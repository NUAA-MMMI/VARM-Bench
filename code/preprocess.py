"""Privacy-safe preprocessing and release auditing for VARM-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata
import re
from typing import Any, Mapping, Sequence


PUBLIC_FIELDS = (
    "text",
    "target",
    "target_type",
    "target_explicitness",
    "stance",
    "label",
    "category",
    "cot",
)

PRIVATE_TEXT_PATTERNS = {
    "email": re.compile(
        r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
    ),
    "url": re.compile(r"(?i)\b(?:https?|git)://[^\s<>]+"),
    "at-handle": re.compile(r"(?<!\w)@[A-Za-z0-9_\-\u4e00-\u9fff]{2,}"),
    "Chinese mobile number": re.compile(
        r"(?<![0-9A-Za-z.])1[3-9]\d{9}(?![0-9A-Za-z.])"
    ),
    "Chinese national ID": re.compile(
        r"(?<![0-9A-Za-z.])\d{17}[\dXx](?![0-9A-Za-z.])"
    ),
    "platform-like numeric ID": re.compile(r"(?<!\d)\d{9,20}(?!\d)"),
}

OFFICIAL_SPLIT_COUNTS = {"train": 5600, "dev": 800, "test": 1600}


def normalize_prompt_input(value: Any) -> str:
    """Collapse whitespace exactly as done before prompt interpolation."""

    return " ".join(str(value or "").split()).strip()


def strong_duplicate_key(value: Any) -> str:
    """Return the NFKC/lower/alphanumeric key used only for deduplication."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(character for character in normalized if character.isalnum())


def project_public_record(row: Mapping[str, Any]) -> dict[str, str]:
    """Validate a reviewed candidate and retain only the eight public fields."""

    missing = [field for field in PUBLIC_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing public fields: {', '.join(missing)}")

    text = normalize_prompt_input(row["text"])
    if len(text) < 15:
        raise ValueError("candidate text must contain at least 15 characters")
    for name, pattern in PRIVATE_TEXT_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"unresolved private identifier pattern: {name}")

    public = {
        field: str(row[field] if row[field] is not None else "").strip()
        for field in PUBLIC_FIELDS
    }
    public["text"] = text
    return public


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_release(
    release_dir: Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Read-only audit of public fields, normalization, privacy, and leakage."""

    expected = dict(expected_counts or OFFICIAL_SPLIT_COUNTS)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    keys_by_split: dict[str, set[str]] = {}

    for split in ("train", "dev", "test"):
        path = release_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        rows_by_split[split] = rows
        if len(rows) != expected[split]:
            errors.append(
                f"{split} count is {len(rows)}; expected {expected[split]}"
            )
        keys: set[str] = set()
        for index, row in enumerate(rows):
            if set(row) != set(PUBLIC_FIELDS):
                errors.append(f"{split}:{index} does not have exactly eight public fields")
                continue
            try:
                projected = project_public_record(row)
            except ValueError as exc:
                errors.append(f"{split}:{index}: {exc}")
                continue
            keys.add(strong_duplicate_key(projected["text"]))
        keys_by_split[split] = keys

    cross_split_overlap = {
        "train_dev": len(keys_by_split["train"] & keys_by_split["dev"]),
        "train_test": len(keys_by_split["train"] & keys_by_split["test"]),
        "dev_test": len(keys_by_split["dev"] & keys_by_split["test"]),
    }
    total_rows = sum(len(rows) for rows in rows_by_split.values())
    normalized_unique = len(set().union(*keys_by_split.values()))
    if normalized_unique != total_rows:
        errors.append(
            f"normalized unique count is {normalized_unique}; rows are {total_rows}"
        )
    if any(cross_split_overlap.values()):
        errors.append(f"cross-split normalized overlap: {cross_split_overlap}")

    near_duplicate_rows_remaining: int | None = None
    report_path = release_dir / "build_report.json"
    if report_path.exists():
        build_report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        near_duplicate_rows_remaining = int(
            build_report.get("near_duplicate_rows_remaining", -1)
        )
        if near_duplicate_rows_remaining != 0:
            errors.append(
                "build report does not certify zero remaining near duplicates"
            )

    return {
        "ok": not errors,
        "release_dir": str(release_dir.resolve()),
        "rows": total_rows,
        "split_counts": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "normalized_unique": normalized_unique,
        "cross_split_overlap": cross_split_overlap,
        "near_duplicate_rows_remaining": near_duplicate_rows_remaining,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit privacy-safe preprocessing of a VARM-Bench release."
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_release(args.release_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
