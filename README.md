# sber

Production-minded factual hallucination detector for the Sber hackathon.

## Final Detector

Final default detector:
- structural features
- base token uncertainty only
- no extra uncertainty groups
- no internal probe features in the default path

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
- lightweight logistic regression head
- output: `hallucination_probability` in `[0, 1]`

Research-only paths kept behind flags:
- extra token uncertainty groups
- internal probe features from a small hidden-layer subset

## Commands

Kaggle / real-provider command sequence:

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
```

## Ablation Summary

Final selection:
- keep `structural + base token uncertainty` as the default detector
- drop extra uncertainty groups from the default path
- drop internal probe features from the default path

Decision rationale:
- base token uncertainty matched the best PR-AUC on the real GigaChat-backed slice
- extra uncertainty groups increased latency without measurable gain
- internal probe features did not justify the added cost on the current slice

## Layout

- `src/` application and research code
- `tests/` unit, integration, contract, and regression checks
- `configs/` runtime and experiment configuration
- `scripts/` automation helpers
- `experiments/` exploratory work
- `artifacts/` generated outputs
- `.codex/` local Codex configuration, skills, and agent role definitions
