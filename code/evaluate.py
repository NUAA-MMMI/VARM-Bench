"""Compute field accuracy, JREM, HER-C, and HER-L."""

from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

FIELDS = ("target_type", "target_explicitness", "stance", "label", "category")


def normalize(value: str) -> str:
    return "".join(str(value).lower().split())


def target_f1(gold: str, predicted: str) -> float:
    left, right = Counter(normalize(gold)), Counter(normalize(predicted))
    overlap = sum((left & right).values())
    if not left and not right:
        return 1.0
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(right.values()), overlap / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def predicted_value(row: dict, field: str) -> object:
    """Read either the public ``predicted_*`` schema or parser-native fields."""

    return row.get(f"predicted_{field}", row.get(field))


def evaluate(gold: list[dict], predicted: list[dict]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("row count mismatch")
    if not gold:
        raise ValueError("gold file is empty")
    field_correct = Counter()
    joint = category_correct = label_correct = 0
    for reference, output in zip(gold, predicted, strict=True):
        if reference["text"] != output["text"]:
            raise ValueError("text alignment mismatch")
        schema_valid = bool(output.get("schema_valid"))
        target_ok = schema_valid and target_f1(
            reference["target"], str(predicted_value(output, "target") or "")
        ) >= 0.5
        checks = {
            field: schema_valid and reference[field] == predicted_value(output, field)
            for field in FIELDS
        }
        field_correct["target"] += target_ok
        for field, ok in checks.items():
            field_correct[field] += ok
        category_correct += checks["category"]
        label_correct += checks["label"]
        joint += target_ok and all(checks.values())
    n = len(gold)
    return {
        "rows": n,
        "field_accuracy": {field: field_correct[field] / n for field in ("target",) + FIELDS},
        "joint_record_match": joint / n,
        "hidden_record_error_given_category": (category_correct - joint) / category_correct if category_correct else 0.0,
        "hidden_record_error_given_label": (label_correct - joint) / label_correct if label_correct else 0.0,
    }


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read(args.gold), read(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
