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
                "positive_count": 14,
                "negative_count": 14,
                "corruption_taxonomy": {"number_nearby": 3, "entity_swap": 3},
            },
            "public_benchmark": {
                "before": {
                    "pr_auc": 0.5938,
                    "false_positive_count": 35,
                    "false_negative_count": 478,
                },
                "after": {
                    "pr_auc": 0.6117,
                    "false_positive_count": 37,
                    "false_negative_count": 470,
                },
                "bucket_deltas": {
                    "numbers": {"false_positive_delta": 1, "false_negative_delta": -4},
                    "entity_like_tokens": {"false_positive_delta": 2, "false_negative_delta": -5},
                    "long_responses": {"false_positive_delta": -1, "false_negative_delta": -6},
                },
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
    assert "before_pr_auc=0.5938" in output
    assert "after_pr_auc=0.6117" in output
    assert "numbers" in output
    assert "artifact=" in output
