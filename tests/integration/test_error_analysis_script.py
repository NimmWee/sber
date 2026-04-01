from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.error_analysis import ErrorBucketSummary


def _load_script_module():
    script_path = PROJECT_ROOT / "scripts" / "run_error_analysis_real_provider.py"
    spec = importlib.util.spec_from_file_location(
        "run_error_analysis_real_provider",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_error_analysis_script_uses_public_benchmark_when_dataset_path_is_given(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    module = _load_script_module()

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"
        response_delimiter = "\n\n### Response:\n"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    dataset_path = tmp_path / "knowledge_bench_public.csv"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "build_ablation_examples",
        lambda: (["train-a", "train-b"], ["unused-validation"]),
    )
    monkeypatch.setattr(module, "resolve_public_benchmark_path", lambda **_: dataset_path)
    monkeypatch.setattr(module, "load_public_benchmark_examples", lambda *_: ["public-a", "public-b"])

    class FakeRunner:
        def __init__(self, *, dataset, artifact_dir) -> None:
            self.dataset = dataset
            self.artifact_dir = artifact_dir

        def run(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                pr_auc=0.6117,
                sample_size=1044,
                false_positive_count=37,
                false_negative_count=477,
                non_trivial_buckets=["numbers", "places"],
                focused_bucket_summaries={
                    "numbers": {"false_positive_count": 2, "false_negative_count": 40},
                    "entity_like_tokens": {"false_positive_count": 5, "false_negative_count": 90},
                    "places": {"false_positive_count": 10, "false_negative_count": 120},
                    "short_responses": {"false_positive_count": 8, "false_negative_count": 70},
                    "long_responses": {"false_positive_count": 12, "false_negative_count": 157},
                },
                recommended_next_improvement="Expand non-public entity and place supervision before adding new model features.",
                hardest_examples=[],
                model_artifact_path=str(artifact_dir / "default_detector_head.json"),
                summary_artifact_path=str(artifact_dir / "error_analysis_summary.json"),
            )

    monkeypatch.setattr(module, "DefaultDetectorErrorAnalysisRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "write_json_artifact",
        lambda **_: artifact_dir / "error_analysis_real_provider_summary.json",
    )
    monkeypatch.setattr(
        module,
        "RawExampleEvaluationDataset",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_error_analysis_real_provider.py",
            "--dataset-path",
            str(dataset_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    module.main()

    captured = capsys.readouterr().out
    assert "sample_size=1044" in captured
    assert "false_positives=37" in captured
    assert "numbers" in captured
    assert "recommended_next_improvement=" in captured


def test_run_error_analysis_script_writes_json_safe_focused_bucket_payload(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_script_module()

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"
        response_delimiter = "\n\n### Response:\n"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    dataset_path = tmp_path / "knowledge_bench_public.csv"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "build_ablation_examples",
        lambda: (["train-a", "train-b"], ["unused-validation"]),
    )
    monkeypatch.setattr(module, "resolve_public_benchmark_path", lambda **_: dataset_path)
    monkeypatch.setattr(module, "load_public_benchmark_examples", lambda *_: ["public-a", "public-b"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_error_analysis_real_provider.py",
            "--dataset-path",
            str(dataset_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    class FakeRunner:
        def __init__(self, *, dataset, artifact_dir) -> None:
            self.dataset = dataset
            self.artifact_dir = artifact_dir

        def run(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                pr_auc=0.6117,
                sample_size=1044,
                false_positive_count=37,
                false_negative_count=477,
                non_trivial_buckets=["numbers", "places"],
                focused_bucket_summaries={
                    "numbers": ErrorBucketSummary(
                        total_count=100,
                        false_positive_count=2,
                        false_negative_count=40,
                    ),
                    "places": ErrorBucketSummary(
                        total_count=200,
                        false_positive_count=10,
                        false_negative_count=120,
                    ),
                    "entity_like_tokens": ErrorBucketSummary(
                        total_count=150,
                        false_positive_count=5,
                        false_negative_count=90,
                    ),
                    "short_responses": ErrorBucketSummary(
                        total_count=120,
                        false_positive_count=8,
                        false_negative_count=70,
                    ),
                    "long_responses": ErrorBucketSummary(
                        total_count=474,
                        false_positive_count=12,
                        false_negative_count=157,
                    ),
                },
                recommended_next_improvement="Expand non-public entity and place supervision before adding new model features.",
                hardest_examples=[],
                model_artifact_path=str(artifact_dir / "default_detector_head.json"),
                summary_artifact_path=str(artifact_dir / "error_analysis_summary.json"),
            )

    monkeypatch.setattr(module, "DefaultDetectorErrorAnalysisRunner", FakeRunner)

    module.main()

    payload = json.loads(
        (artifact_dir / "error_analysis_real_provider_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["focused_bucket_summaries"]["numbers"]["false_negative_count"] == 40
