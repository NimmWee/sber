from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module():
    script_path = PROJECT_ROOT / "scripts" / "run_non_public_recovery_public_eval.py"
    spec = importlib.util.spec_from_file_location(
        "run_non_public_recovery_public_eval",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_non_public_recovery_script_prints_dataset_and_before_after_metrics(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    module = _load_script_module()

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    public_path = tmp_path / "knowledge_bench_public.csv"
    artifact_dir = tmp_path / "artifacts"
    baseline_model_path = tmp_path / "baseline_lightgbm_head.json"

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "resolve_public_benchmark_path", lambda **_: public_path)
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "run_non_public_retraining_public_eval",
        lambda **_: {
            "dataset_summary": {
                "sample_size": 28,
                "non_hallucination_count": 18,
                "hallucination_count": 10,
                "effective_label_balance": {"hallucination_ratio": 0.42},
                "corruption_taxonomy": {"number_nearby": 3, "entity_swap": 3},
            },
            "public_benchmark": {
                "before": {
                    "pr_auc": 0.5938,
                    "false_positive_count": 35,
                    "false_negative_count": 478,
                    "precision": 0.50,
                    "recall": 0.22,
                    "predicted_positive_rate": 0.30,
                    "score_distribution": {
                        "overall": {"mean": 0.31, "q10": 0.05, "q50": 0.24, "q90": 0.72},
                        "hallucination": {"mean": 0.48, "q10": 0.12, "q50": 0.43, "q90": 0.84},
                        "non_hallucination": {"mean": 0.18, "q10": 0.03, "q50": 0.11, "q90": 0.42},
                    },
                },
                "after": {
                    "pr_auc": 0.6117,
                    "false_positive_count": 37,
                    "false_negative_count": 470,
                    "precision": 0.52,
                    "recall": 0.24,
                    "predicted_positive_rate": 0.31,
                    "score_distribution": {
                        "overall": {"mean": 0.33, "q10": 0.06, "q50": 0.28, "q90": 0.75},
                        "hallucination": {"mean": 0.50, "q10": 0.14, "q50": 0.47, "q90": 0.86},
                        "non_hallucination": {"mean": 0.19, "q10": 0.03, "q50": 0.12, "q90": 0.43},
                    },
                },
                "bucket_deltas": {
                    "numbers": {"false_positive_delta": 1, "false_negative_delta": -4},
                    "entity_like_tokens": {"false_positive_delta": 2, "false_negative_delta": -5},
                    "places": {"false_positive_delta": 0, "false_negative_delta": -2},
                    "long_responses": {"false_positive_delta": -1, "false_negative_delta": -6},
                },
            },
            "recall_recovery": {
                "false_negatives_decreased": True,
                "false_negative_delta": -8,
                "false_positive_increase": 2,
                "false_positive_increase_too_much": False,
            },
            "precision_change": 0.02,
            "guardrails": {
                "predicted_positive_rate_drop_too_far": False,
                "recall_collapsed": False,
                "long_response_positive_rate_collapsed": False,
                "score_distribution_compressed": False,
            },
            "training_config": {
                "recovery_blend_weight": 0.35,
                "train_sample_weight_sum": 21.5,
            },
            "decision": {
                "accept_change": True,
                "rejection_reason": None,
                "false_positive_limit": 42,
                "false_positive_increase_too_much": False,
            },
            "artifact_path": str(artifact_dir / "non_public_recovery_summary.json"),
            "trained_model_artifact_path": str(artifact_dir / "retrained_default_detector_head.json"),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_non_public_recovery_public_eval.py",
            "--dataset-path",
            str(public_path),
            "--artifact-dir",
            str(artifact_dir),
            "--baseline-model-artifact-path",
            str(baseline_model_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "dataset_size=28" in output
    assert "non_hallucination_count=18" in output
    assert "hallucination_count=10" in output
    assert "effective_hallucination_ratio=0.4200" in output
    assert "before_pr_auc=0.5938" in output
    assert "after_pr_auc=0.6117" in output
    assert "before_recall=0.2200" in output
    assert "after_recall=0.2400" in output
    assert "before_predicted_positive_rate=0.3000" in output
    assert "after_predicted_positive_rate=0.3100" in output
    assert "before_score_mean_hallucination=0.4800" in output
    assert "after_score_mean_hallucination=0.5000" in output
    assert "numbers" in output
    assert "false_negatives_decreased=True" in output
    assert "accept_change=True" in output
    assert "artifact=" in output
