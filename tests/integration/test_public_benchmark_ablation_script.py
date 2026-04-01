from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module():
    script_path = PROJECT_ROOT / "scripts" / "run_public_benchmark_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "run_public_benchmark_ablation",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_public_benchmark_ablation_script_prints_variant_summary(
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

    dataset_path = tmp_path / "knowledge_bench_public.csv"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr(module, "resolve_public_benchmark_path", lambda **_: dataset_path)
    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "build_ablation_examples",
        lambda: (["train-example"], ["validation-example"]),
    )
    monkeypatch.setattr(
        module,
        "run_public_benchmark_ablation",
        lambda **_: {
            "sample_size": 1044,
            "best_variant": "base_token_uncertainty",
            "artifact_path": str(artifact_dir / "public_benchmark_ablation_summary.json"),
            "variants": {
                "base_token_uncertainty": {
                    "pr_auc": 0.5152,
                    "latency_total_mean_ms": 24.1,
                    "false_positive_count": 1,
                    "false_negative_count": 566,
                    "non_trivial_buckets": ["entity_like_tokens", "places"],
                },
                "extended_token_uncertainty": {
                    "pr_auc": 0.5100,
                    "latency_total_mean_ms": 25.7,
                    "precision": 0.49,
                    "recall": 0.18,
                    "predicted_positive_rate": 0.13,
                    "false_positive_count": 2,
                    "false_negative_count": 560,
                    "non_trivial_buckets": ["entity_like_tokens"],
                },
                "fused_specialist_ensemble": {
                    "pr_auc": 0.5320,
                    "latency_total_mean_ms": 28.3,
                    "precision": 0.53,
                    "recall": 0.21,
                    "predicted_positive_rate": 0.14,
                    "false_positive_count": 3,
                    "false_negative_count": 548,
                    "non_trivial_buckets": ["numbers", "long_responses"],
                },
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_public_benchmark_ablation.py",
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    module.main()

    captured = capsys.readouterr()
    assert "dataset=" in captured.out
    assert "model=/kaggle/temp/GigaChat3" in captured.out
    assert "best_variant=base_token_uncertainty" in captured.out
    assert "base_token_uncertainty pr_auc=0.5152" in captured.out
    assert "extended_token_uncertainty pr_auc=0.5100" in captured.out
    assert "fused_specialist_ensemble pr_auc=0.5320" in captured.out
    assert "artifact=" in captured.out
