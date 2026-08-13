# VARM-Bench

Reference code for **VARM-Bench: Benchmarking Verifiable Structured Reasoning in Chinese Abusive Speech Moderation**.

This code-only repository contains the six-anchor protocol, deterministic parser, record validator, privacy-safe preprocessing checks, and evaluation code for JREM, HER-C, HER-L, and the lexical-cue subsets. It excludes the benchmark data, model checkpoints, full prediction files, paper sources, and internal audit artifacts.

## Requirements

- Python 3.10 or later
- No third-party Python packages for the included scripts

## Quick start

Validate the synthetic reference records:

```bash
python code/validate.py --input examples/gold.jsonl --expected-count 2
```

Parse raw model responses:

```bash
python code/parse_predictions.py \
  --input examples/raw_outputs.jsonl \
  --output outputs/predictions.jsonl
```

Compute the core metrics:

```bash
python code/evaluate.py \
  --gold examples/gold.jsonl \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Record format

Each reference JSONL row has eight fields:

```text
text, target, target_type, target_explicitness, stance, label, category, cot
```

The rationale contains exactly six ordered anchors:

```text
[T:] -> [TY:] -> [TT:] -> [S:] -> [L:] -> [C:]
```

See `prompts/protocol.txt` for the exact zero-shot, taxonomy-guided, and CoT-SFT prompt contract.

## Evaluation

The target matches when normalized character-overlap F1 is at least 0.5. Target type, target explicitness, stance, label, and category use exact matching. JREM requires the target and all five discrete fields to match. HER-C and HER-L report hidden complete-record errors among category-correct and label-correct outputs.

`code/evaluate_q2.py` applies the same evaluator to cue-conditioned subset files. It expects one or more prediction JSONL files under `--prediction-root` and subset JSONL files under `--subset-dir`.

## Data

The benchmark data is not included in this core-code release. The files under `examples/` are synthetic and exist only for testing the public interface.

## Citation

Citation metadata is available in `CITATION.cff`.
