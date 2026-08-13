from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluate import evaluate, target_f1  # noqa: E402
from parse_predictions import convert  # noqa: E402
from protocol import parse_prediction, validate_record  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = read_jsonl(ROOT / "examples" / "gold.jsonl")
        cls.raw = read_jsonl(ROOT / "examples" / "raw_outputs.jsonl")

    def test_reference_records_validate(self) -> None:
        for row in self.gold:
            validate_record(row)

    def test_parser_rejects_anchor_only_output(self) -> None:
        parsed = parse_prediction(
            "分析：[T:你][TY:单一对象][TT:明示][S:攻击或认同][L:有害][C:一般辱骂]"
        )
        self.assertFalse(parsed["schema_valid"])
        self.assertIn("anchors_not_naturally_integrated", parsed["format_issues"])

    def test_raw_conversion_and_evaluation(self) -> None:
        predictions = [convert(row) for row in self.raw]
        result = evaluate(self.gold, predictions)
        self.assertEqual(result["joint_record_match"], 1.0)
        self.assertEqual(result["hidden_record_error_given_category"], 0.0)
        self.assertEqual(result["hidden_record_error_given_label"], 0.0)

    def test_target_f1(self) -> None:
        self.assertEqual(target_f1("某个对象", "某个对象"), 1.0)
        self.assertGreater(target_f1("这部电影", "电影"), 0.5)


if __name__ == "__main__":
    unittest.main()
