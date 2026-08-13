"""Evaluate compact predictions on the four cue-conditioned subsets."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from evaluate import evaluate


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=Path("data/subsets/q2"))
    parser.add_argument("--prediction-root", type=Path, default=Path("predictions"))
    parser.add_argument("--output", type=Path, default=Path("outputs/q2.json"))
    args = parser.parse_args()
    all_predictions = list(args.prediction_root.rglob("*.jsonl"))
    subset_paths = sorted(args.subset_dir.glob("*.jsonl"))
    if not all_predictions:
        raise ValueError(f"no prediction files found under {args.prediction_root}")
    if not subset_paths:
        raise ValueError(f"no subset files found under {args.subset_dir}")
    scores = []
    for prediction_path in sorted(all_predictions):
        prediction_rows = read(prediction_path)
        by_text = {row["text"]: row for row in prediction_rows}
        for subset_path in subset_paths:
            gold = read(subset_path)
            scores.append({"system": prediction_path.relative_to(args.prediction_root).as_posix(), "subset": subset_path.stem, **evaluate(gold, [by_text[row["text"]] for row in gold])})
    payload = {"prediction_files": len(all_predictions), "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_files": len(all_predictions), "score_rows": len(scores)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
