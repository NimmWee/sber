# sber

Production-grade factual hallucination detector for the Sber hackathon.

## Frozen Submission Candidate

The frozen final submission candidate is the exact historical best-performing build:
- historical best commit: `d3fa946`
- frozen variant: `baseline_plus_all_specialists`
- historical public PR-AUC: `0.6881`

This repository now treats that build as the only active submission path.

## Method Summary

Input:
- `prompt + response`

Runtime scoring path:
- one forward pass of local GigaChat on `prompt + delimiter + response`
- structural features
- base token uncertainty
- compact internal features
- lightweight specialist scorers:
  - numeric
  - entity-like
  - long-response
- frozen weighted blend:
  - baseline `0.55`
  - numeric specialist `0.15`
  - entity specialist `0.15`
  - long specialist `0.15`

Output:
- `hallucination_probability` in `[0, 1]`

## Dataset And Provenance

Training data is text-based and auditable.

The repository preserves:
- public factual seeds in `data/public_seed_facts.jsonl`
- textual dataset builder in `src/data/textual_dataset.py`
- feature reconstruction from text in `src/data/textual_preprocessing.py`
- frozen submission training/scoring code in `src/submission/frozen_best.py`

The preview benchmark is for evaluation only and is not used as training data.
It is referenced only for overlap and leakage checks during dataset construction.
The committed public preview file lives at `data/bench/knowledge_bench_public.csv`.

## Reproducible Commands

Install:
```bash
bash scripts/install.sh
```

Train the frozen submission candidate:
```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

Score the private benchmark:
```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

Explicit Python entrypoints:
```bash
python scripts/build_text_training_dataset.py
python scripts/train_frozen_submission.py --config configs/token_stat_provider.local.json --dataset-path data/processed/textual_training_dataset.jsonl --artifact-dir model/frozen_best
python scripts/score_frozen_submission.py --config configs/token_stat_provider.local.json --input-path data/bench/knowledge_bench_private.csv --artifact-dir model/frozen_best --output-path data/bench/knowledge_bench_private_scores.csv
```

## Expected Files

Private scoring input:
- `data/bench/knowledge_bench_private.csv`

Private scoring output:
- `data/bench/knowledge_bench_private_scores.csv`

Frozen model artifacts:
- `model/frozen_best/`

Frozen submission config:
- `configs/frozen_submission.json`

## Repository Layout

- `configs/` runtime and frozen-submission configuration
- `data/bench/` private benchmark input/output location
- `data/bench/` benchmark CSVs:
  - `knowledge_bench_public.csv` for evaluation-only overlap checks
  - `knowledge_bench_private.csv` as submission input
  - `knowledge_bench_private_scores.csv` as submission output
- `data/` text-based seed data and processed datasets
- `model/frozen_best/` frozen submission artifacts
- `src/submission/` final frozen submission code
- `scripts/` install, train, score, and utility entrypoints
- `scripts/` active executable surface:
  - `install.sh`
  - `train.sh`
  - `score_private.sh`
  - `build_text_training_dataset.py`
  - `preprocess_text_training_dataset.py`
  - `train_frozen_submission.py`
  - `score_frozen_submission.py`
- `notebooks/` optional analysis notebooks

## Reproducibility Notes

- The active submission path is frozen to the historical best specialist blend.
- Runtime scoring does not use external APIs, retrieval, or multi-pass generation.
- `scripts/install.sh` prepares directories but does not create private benchmark inputs.
- Manual setup is required for:
  - `data/bench/knowledge_bench_private.csv`
  - `configs/token_stat_provider.local.json` pointing to the local GigaChat checkpoint
- If the textual training dataset has not been built yet, `scripts/train.sh` rebuilds it from the committed public seed inputs.
