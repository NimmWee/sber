# sber

Production-minded factual hallucination detector for the Sber hackathon.

## Final Detector

Final default detector:
- structural features
- base token uncertainty only
- compact internal features
- LightGBM head
- no extra uncertainty groups in the default path
- specialist ensemble variants kept for research / ablation only

Current default token uncertainty set:
- `token_mean_logprob`
- `token_min_logprob`
- `token_entropy_mean`
- `token_top1_top2_margin_mean`
- `token_tail_low_confidence_rate`
- `token_confidence_decay`

## Architecture

Runtime path:
- single forward pass over `prompt + delimiter + response`
- token-stat collection for response tokens only
- deterministic feature extraction
- lightweight LightGBM head
- output: `hallucination_probability` in `[0, 1]`

Research-only paths kept behind flags:
- extra token uncertainty groups
- specialist ensemble / fusion comparisons

Training data path:
- text-based, auditable training examples with provenance
- public seed ingestion + synthetic corruptions + correct factual augmentations
- benchmark used for evaluation only

## Reproducible Commands

Install:

```bash
bash scripts/install.sh
```

Build textual training dataset:

```bash
python scripts/build_text_training_dataset.py
```

Preprocess training dataset into model-ready rows:

```bash
python scripts/preprocess_text_training_dataset.py --config configs/token_stat_provider.local.json
```

Train current detector:

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

Score private dataset:

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json --input-path private_test.csv
```

Provider / evaluation commands:

```bash
python scripts/run_smoke_provider.py --config configs/token_stat_provider.local.json
python scripts/run_eval_real_provider.py --config configs/token_stat_provider.local.json
python scripts/run_latency_real_provider.py --config configs/token_stat_provider.local.json
python scripts/run_error_analysis_real_provider.py --config configs/token_stat_provider.local.json
```

Optional research-only comparisons:

```bash
python scripts/run_ablation_real_provider.py --config configs/token_stat_provider.local.json
python scripts/run_internal_probe_compare_real_provider.py --config configs/token_stat_provider.local.json
python scripts/run_public_benchmark_ablation.py --config configs/token_stat_provider.local.json
```

## Ablation Summary

Final selection:
- keep `structural + base token uncertainty + compact internal features + LightGBM` as the strongest default detector family
- drop extra uncertainty groups from the default path
- keep specialist ensemble and recovery-style variants behind explicit comparison paths

Decision rationale:
- token uncertainty provides the main stable signal
- compact internal features improved the strongest full-benchmark result
- larger feature families and recovery-only rebalancing did not generalize reliably enough to become default

## Layout

- `data/` public seed facts and processed textual training datasets
- `model/` trained detector artifacts
- `notebooks/` optional reproducible analysis notebooks
- `src/` application and research code
- `tests/` unit, integration, contract, and regression checks
- `configs/` runtime and experiment configuration
- `scripts/` automation helpers
- `experiments/` exploratory work
- `artifacts/` generated outputs
- `.codex/` local Codex configuration, skills, and agent role definitions
