"""
Post-processing module for check extraction model output.

Takes raw JSON from the model and applies:
- Null normalization (empty strings, "N/A", etc. → null)
- Amount cleanup (strip $, commas, asterisks, "DOLLARS")
- Payor cleanup (strip addresses after newline)
- Fractional number validation (regex: XX-YYYY/ZZZZ)
- Fractional rescue (find fractional in wrong fields, move it)
- ABA routing number calculation from valid fractional

Usage:
    from post_process import post_process

    raw_model_output = {"payorInstitution": "CHASE", "amount": "$5,000.00", ...}
    cleaned = post_process(raw_model_output)
"""

import re

EXPECTED_FIELDS = [
    "payorInstitution", "payor", "payee", "amount",
    "account", "serial", "checkDate", "fractionalNumber",
    "calculatedRoutingNumber",
]

FRACTIONAL_PATTERN = re.compile(r"^\d{1,2}-\d{1,4}/\d{1,4}$")


def normalize_nulls(data):
    for field in EXPECTED_FIELDS:
        val = data.get(field)
        if val is not None and isinstance(val, str):
            stripped = val.strip()
            if stripped == "" or stripped.lower() in ("null", "n/a", "none"):
                data[field] = stripped if stripped == "" else None
                data[field] = None
    return data


def clean_amount(value):
    if not value or not isinstance(value, str):
        return value
    cleaned = value.strip()
    cleaned = cleaned.replace("$", "").replace(",", "").replace("*", "")
    cleaned = re.sub(r"\s*(dollars?|only)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    match = re.match(r"^(\d+\.?\d*)$", cleaned)
    if match:
        num = match.group(1)
        if "." not in num:
            num += ".00"
        return num
    return value


def clean_payor(value):
    if not value or not isinstance(value, str):
        return value
    lines = value.split("\n")
    name = lines[0].strip()
    if not name:
        return None
    return name


def validate_fractional(value):
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if FRACTIONAL_PATTERN.match(cleaned):
        return cleaned
    return None


def extract_fractional_from_string(value):
    if not value or not isinstance(value, str):
        return None
    match = re.search(r"\d{1,2}-\d{1,4}/\d{1,4}", value)
    if match:
        return match.group()
    return None


def calculate_routing_number(fractional):
    if not fractional:
        return None
    match = re.match(r"(\d{1,2})-(\d{1,4})/(\d{1,4})", fractional)
    if not match:
        return None

    bank_id = match.group(2).zfill(4)
    fed_routing = match.group(3).zfill(4)
    base = fed_routing + bank_id

    digits = [int(d) for d in base]
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    weighted = sum(d * w for d, w in zip(digits, weights))
    check_digit = (10 - (weighted % 10)) % 10

    return f"{base}{check_digit}"


def ensure_fields(data):
    for field in EXPECTED_FIELDS:
        if field not in data:
            data[field] = None
    return data


def post_process(data):
    ensure_fields(data)
    normalize_nulls(data)

    if data.get("amount"):
        data["amount"] = clean_amount(data["amount"])

    if data.get("payor"):
        data["payor"] = clean_payor(data["payor"])

    frn = data.get("fractionalNumber")
    validated = validate_fractional(frn)

    if not validated:
        for field in EXPECTED_FIELDS:
            if field in ("fractionalNumber", "calculatedRoutingNumber"):
                continue
            found = extract_fractional_from_string(data.get(field))
            if found:
                validated = found
                data[field] = None
                break

    if frn and not validated:
        data["_fractionalRaw"] = frn

    data["fractionalNumber"] = validated
    data["calculatedRoutingNumber"] = calculate_routing_number(validated)

    return data


if __name__ == "__main__":
    import json
    import sys

    example = {
        "payorInstitution": "J.P.MORGAN CHASE BANK",
        "payor": "John Smith\n123 Main St\nNew York, NY 10001",
        "payee": "Jane Doe",
        "amount": "$5,000.00",
        "account": "",
        "serial": "8381",
        "checkDate": "2026-04-06",
        "fractionalNumber": "87-176/843",
    }

    print("Input:")
    print(json.dumps(example, indent=2))
    print("\nOutput:")
    print(json.dumps(post_process(example), indent=2))
