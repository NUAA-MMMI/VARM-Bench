"""Validate VARM-Bench reference records and their anchored rationales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import FIELDS, validate_record


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def validate_file(path: Path, expected_count: int | None = None) -> dict:
    rows = read_rows(path)
    errors: list[str] = []
    seen_texts: set[str] = set()

    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"row count is {len(rows)}; expected {expected_count}")

    for index, row in enumerate(rows):
        if set(row) != set(FIELDS):
            missing = sorted(set(FIELDS) - set(row))
            extra = sorted(set(row) - set(FIELDS))
            errors.append(f"row {index}: missing={missing}, extra={extra}")
            continue
        if row["text"] in seen_texts:
            errors.append(f"row {index}: duplicate text")
        seen_texts.add(row["text"])
        try:
            validate_record(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")

    return {"ok": not errors, "rows": len(rows), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a VARM-Bench reference JSONL file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    result = validate_file(args.input, args.expected_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
