"""
Fine-tune Qwen2.5-VL-7B for bank check field extraction using Unsloth + QLoRA.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate

Usage:
    python train.py

Expects this directory structure:
    training_data/
        images/          check_00025.jpg, check_00028.jpg, ...
        labels/          check_00025.json, check_00028.json, ...
        holdout/
            images/      check_00456.jpg, ...
            labels/      check_00456.json, ...

Outputs:
    output/
        check-extractor-q4km.gguf    (~4.7GB quantized model)
        Modelfile                     (for ollama create)
        adapter/                      (LoRA adapter backup)
        baseline_eval.json
        post_training_eval.json
"""

import json
import base64
import os
import sys
import re
import time
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct"
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
GRAD_ACCUM = 4
LORA_RANK = 16
MAX_SEQ_LEN = 2048
EVAL_SAMPLES = 50  # How many holdout samples to evaluate (set to None for all)

BASE_DIR = Path(__file__).parent
TRAIN_IMAGES = BASE_DIR / "training_data" / "images"
TRAIN_LABELS = BASE_DIR / "training_data" / "labels"
HOLDOUT_IMAGES = BASE_DIR / "training_data" / "holdout" / "images"
HOLDOUT_LABELS = BASE_DIR / "training_data" / "holdout" / "labels"
OUTPUT_DIR = BASE_DIR / "output"

SYSTEM_PROMPT = """You are a bank check data extraction model. Given an image of a check, extract the following fields and return ONLY a valid JSON object with no markdown, no code fences, no explanation. Use null (not empty string) for any field you cannot find or read.

If any text is partially covered, obscured, redacted, or not fully readable, set that field to null. Do NOT guess or reconstruct partially visible text.

Fields to extract:
- payorInstitution: The name of the bank printed on the check (e.g. "BANK OF AMERICA")
- payor: The account holder who wrote the check (name ONLY, no address)
- payee: The person or entity on the "Pay to the order of" line
- amount: Dollar amount as a plain number with two decimal places (e.g. "2675.00"). No dollar signs, commas, or words.
- account: Account number from the MICR line at the bottom of the check
- serial: Check serial/number (typically top-right or MICR line)
- checkDate: Date written on the check in YYYY-MM-DD format
- fractionalNumber: The fractional routing number near the check number, format: DD-DDD/DDDD (e.g. "87-176/843"). This is NOT the 9-digit routing number.
- calculatedRoutingNumber: Leave this as null — it will be computed in post-processing."""

USER_PROMPT = "Extract all fields from this bank check image and return a JSON object."

EXPECTED_FIELDS = [
    "payorInstitution", "payor", "payee", "amount",
    "account", "serial", "checkDate", "fractionalNumber",
]


# ─── Step 1: Verify Environment ──────────────────────────────────────
def verify_environment():
    print("=" * 60)
    print("STEP 1: ENVIRONMENT CHECK")
    print("=" * 60)

    import torch
    assert torch.cuda.is_available(), "CUDA not available — need an NVIDIA GPU"
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import unsloth
    print(f"  Unsloth: {unsloth.__version__}")

    assert TRAIN_IMAGES.exists(), f"Training images not found: {TRAIN_IMAGES}"
    assert TRAIN_LABELS.exists(), f"Training labels not found: {TRAIN_LABELS}"

    train_count = len(list(TRAIN_IMAGES.glob("*.jpg"))) + len(list(TRAIN_IMAGES.glob("*.png")))
    label_count = len(list(TRAIN_LABELS.glob("*.json")))
    print(f"  Training: {train_count} images, {label_count} labels")

    holdout_count = 0
    if HOLDOUT_IMAGES.exists():
        holdout_count = len(list(HOLDOUT_IMAGES.glob("*.jpg"))) + len(list(HOLDOUT_IMAGES.glob("*.png")))
        holdout_labels = len(list(HOLDOUT_LABELS.glob("*.json")))
        print(f"  Holdout: {holdout_count} images, {holdout_labels} labels")

    assert train_count >= 50, f"Need at least 50 training images, found {train_count}"
    print("\n  PASSED\n")
    return train_count, holdout_count


# ─── Step 2: Build Dataset ───────────────────────────────────────────
def clean_label(label):
    label.pop("source_pool", None)
    label["calculatedRoutingNumber"] = None

    payor = label.get("payor")
    if payor and isinstance(payor, str):
        label["payor"] = payor.split("\n")[0].strip() or None

    return label


from torch.utils.data import Dataset
from PIL import Image


class CheckDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.samples = []
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        for img_path in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))):
            label_path = label_dir / (img_path.stem + ".json")
            if not label_path.exists():
                continue

            with open(label_path) as f:
                label = clean_label(json.load(f))

            self.samples.append({
                "image_path": str(img_path),
                "ground_truth": json.dumps(label, indent=2),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["ground_truth"]}],
                },
            ]
        }


# ─── Step 3: Evaluation ──────────────────────────────────────────────
def parse_json_output(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def run_inference(image_path, model, tokenizer):
    import torch
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        text=[text_input], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=512, temperature=0.1,
            do_sample=False, repetition_penalty=1.1,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, image_dir, label_dir, max_samples=None):
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if max_samples:
        image_files = image_files[:max_samples]

    field_correct = {f: 0 for f in EXPECTED_FIELDS}
    field_total = {f: 0 for f in EXPECTED_FIELDS}
    results = []

    for i, img_path in enumerate(image_files):
        label_path = label_dir / (img_path.stem + ".json")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            truth = clean_label(json.load(f))

        raw = run_inference(str(img_path), model, tokenizer)
        predicted = parse_json_output(raw) or {}

        sample_result = {"image": img_path.name, "predicted": predicted, "truth": truth, "fields": {}}

        for field in EXPECTED_FIELDS:
            t = truth.get(field)
            p = predicted.get(field)
            t_str = str(t).strip().lower() if t else ""
            p_str = str(p).strip().lower() if p else ""

            if not t_str:
                continue
            field_total[field] += 1
            if t_str == p_str:
                field_correct[field] += 1
                sample_result["fields"][field] = "correct"
            else:
                sample_result["fields"][field] = f"mismatch (gt={t!r}, pred={p!r})"

        results.append(sample_result)

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(image_files)}...")

    total_c = sum(field_correct.values())
    total_t = sum(field_total.values())
    accuracy = (total_c / total_t * 100) if total_t > 0 else 0

    print(f"\n  Per-field accuracy:")
    for field in EXPECTED_FIELDS:
        c, t = field_correct[field], field_total[field]
        pct = (c / t * 100) if t > 0 else 0
        print(f"    {field:30s}: {c:4d}/{t:4d} ({pct:5.1f}%)")
    print(f"\n  Overall: {total_c}/{total_t} ({accuracy:.1f}%)")

    return {"accuracy": accuracy, "field_correct": field_correct, "field_total": field_total, "details": results}


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    start_time = time.time()

    # Step 1: Verify
    train_count, holdout_count = verify_environment()

    # Step 2: Load model
    print("=" * 60)
    print("STEP 2: LOAD MODEL")
    print("=" * 60)

    import torch
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print(f"  Model loaded: {MODEL_NAME}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")

    # Step 3: Apply LoRA
    print("=" * 60)
    print("STEP 3: APPLY LORA")
    print("=" * 60)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
    )
    model.print_trainable_parameters()

    # Step 4: Build dataset
    print("\n" + "=" * 60)
    print("STEP 4: BUILD DATASET")
    print("=" * 60)

    train_dataset = CheckDataset(TRAIN_IMAGES, TRAIN_LABELS)
    print(f"  Training samples: {len(train_dataset)}")

    sample = train_dataset[0]
    print(f"  Sample keys: {sample.keys()}")
    print(f"  Message roles: {[m['role'] for m in sample['messages']]}")

    # Step 5: Baseline evaluation
    print("\n" + "=" * 60)
    print("STEP 5: BASELINE EVALUATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if holdout_count > 0:
        baseline = evaluate(model, tokenizer, HOLDOUT_IMAGES, HOLDOUT_LABELS, max_samples=EVAL_SAMPLES)
        with open(OUTPUT_DIR / "baseline_eval.json", "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"  Saved: {OUTPUT_DIR}/baseline_eval.json")
    else:
        print("  No holdout set — skipping baseline")
        baseline = {"accuracy": 0}

    # Step 6: Train
    print("\n" + "=" * 60)
    print("STEP 6: TRAINING")
    print("=" * 60)

    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        remove_unused_columns=False,
        report_to="none",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    total_steps = len(train_dataset) * TRAIN_EPOCHS // GRAD_ACCUM
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Epochs: {TRAIN_EPOCHS}")
    print(f"  Total optimizer steps: ~{total_steps}")
    print(f"  Starting training...\n")

    trainer_stats = trainer.train()

    runtime = trainer_stats.metrics["train_runtime"]
    print(f"\n  Training complete.")
    print(f"  Runtime: {runtime:.0f}s ({runtime/60:.1f}min)")
    print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")

    # Step 7: Post-training evaluation
    print("\n" + "=" * 60)
    print("STEP 7: POST-TRAINING EVALUATION")
    print("=" * 60)

    if holdout_count > 0:
        post_eval = evaluate(model, tokenizer, HOLDOUT_IMAGES, HOLDOUT_LABELS, max_samples=EVAL_SAMPLES)
        with open(OUTPUT_DIR / "post_training_eval.json", "w") as f:
            json.dump(post_eval, f, indent=2)

        delta = post_eval["accuracy"] - baseline["accuracy"]
        print(f"\n  Baseline: {baseline['accuracy']:.1f}%")
        print(f"  Post-training: {post_eval['accuracy']:.1f}%")
        print(f"  Delta: {delta:+.1f}%")
    else:
        print("  No holdout set — skipping")

    # Step 8: Save adapter
    print("\n" + "=" * 60)
    print("STEP 8: SAVE ADAPTER")
    print("=" * 60)

    adapter_dir = OUTPUT_DIR / "adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  Adapter saved: {adapter_dir}")

    # Step 9: Export GGUF
    print("\n" + "=" * 60)
    print("STEP 9: EXPORT GGUF")
    print("=" * 60)

    gguf_dir = OUTPUT_DIR / "gguf"
    os.makedirs(gguf_dir, exist_ok=True)

    print("  Merging adapter and exporting to GGUF (q4_k_m)...")
    print("  This takes 5-15 minutes. Do not interrupt.\n")

    merged_dir = OUTPUT_DIR / "merged"
    os.makedirs(merged_dir, exist_ok=True)

    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Check if Unsloth actually produced a GGUF or just safetensors
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]

    if not gguf_files:
        print("  Unsloth saved safetensors instead of GGUF — using llama.cpp fallback")
        import subprocess

        llama_cpp = BASE_DIR / "llama.cpp"
        if not llama_cpp.exists():
            print("  Cloning llama.cpp...")
            subprocess.run(["git", "clone", "--depth", "1",
                          "https://github.com/ggml-org/llama.cpp", str(llama_cpp)], check=True)

        print("  Installing llama.cpp Python requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                       str(llama_cpp / "requirements.txt")], check=True)

        # Convert safetensors to f16 GGUF
        f16_path = gguf_dir / "check-extractor-f16.gguf"
        print("  Converting to f16 GGUF...")
        subprocess.run([
            sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"),
            str(gguf_dir), "--outfile", str(f16_path), "--outtype", "f16",
        ], check=True)

        # Build quantize tool
        print("  Building llama.cpp quantize tool...")
        subprocess.run(["cmake", "-B", "build"], cwd=str(llama_cpp), check=True,
                       capture_output=True)
        subprocess.run(["cmake", "--build", "build", "--target", "llama-quantize", "-j4"],
                       cwd=str(llama_cpp), check=True, capture_output=True)

        # Quantize to Q4_K_M
        q4_path = gguf_dir / "check-extractor-q4km.gguf"
        quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
        print("  Quantizing to Q4_K_M...")
        subprocess.run([str(quantize_bin), str(f16_path), str(q4_path), "Q4_K_M"], check=True)

        # Clean up f16 to save space
        if q4_path.exists():
            os.remove(f16_path)
            print(f"  Removed f16 intermediate (kept q4_k_m)")

        # Clean up safetensors from gguf dir
        for f in os.listdir(gguf_dir):
            if f.endswith(".safetensors"):
                os.remove(gguf_dir / f)

        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]

    print("\n  GGUF export complete. Files:")
    for f in os.listdir(gguf_dir):
        fpath = gguf_dir / f
        if fpath.is_file():
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"    {f} ({size_mb:.1f} MB)")

    # Write Modelfile if not auto-generated
    modelfile_path = gguf_dir / "Modelfile"
    if not modelfile_path.exists() and gguf_files:
        with open(modelfile_path, "w") as f:
            f.write(f"FROM ./{gguf_files[0]}\n")
            f.write("PARAMETER temperature 0.1\n")
            f.write(f'SYSTEM """{SYSTEM_PROMPT}"""\n')
        print(f"    Modelfile created")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  GGUF: {gguf_dir}")
    print(f"  To load into Ollama:")
    print(f"    cd {gguf_dir}")
    print(f"    ollama create check-extractor -f Modelfile")
    print("=" * 60)


if __name__ == "__main__":
    main()
