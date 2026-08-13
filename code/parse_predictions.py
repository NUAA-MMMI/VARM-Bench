"""Parse raw model outputs into the compact VARM-Bench prediction schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import parse_prediction


PREDICTED_FIELDS = (
    "target",
    "target_type",
    "target_explicitness",
    "stance",
    "label",
    "category",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def convert(row: dict) -> dict:
    if "text" not in row or "output" not in row:
        raise ValueError("each raw row must contain text and output")
    parsed = parse_prediction(row["output"])
    compact = {
        "text": row["text"],
        "schema_valid": parsed["schema_valid"],
    }
    compact.update(
        {f"predicted_{field}": parsed[field] for field in PREDICTED_FIELDS}
    )
    compact["format_issues"] = parsed["format_issues"]
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse six anchored fields from raw VARM-Bench outputs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    converted = [convert(row) for row in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid = sum(row["schema_valid"] for row in converted)
    print(json.dumps({"rows": len(converted), "schema_valid": valid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
