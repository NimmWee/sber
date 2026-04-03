from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_text_training_dataset_script_prints_summary(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module("build_text_training_dataset.py")
    dataset_path = tmp_path / "dataset.jsonl"

    monkeypatch.setattr(
        module,
        "build_textual_training_dataset",
        lambda **_: type(
            "Dataset",
            (),
            {
                "summary": {
                    "sample_size": 24,
                    "hallucination_count": 10,
                    "non_hallucination_count": 14,
                    "source_name_distribution": {"seed_a": 10, "seed_b": 14},
                    "hallucination_bucket_coverage": {"numbers": 8, "entity_like_tokens": 6},
                    "difficulty_heuristics": {"small_numeric_delta_hallucination_count": 4},
                    "warnings": ["long-response coverage is too low"],
                },
                "train_examples": [],
                "dev_examples": [],
            },
        )(),
    )
    monkeypatch.setattr(module, "export_textual_training_dataset", lambda **_: dataset_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_text_training_dataset.py", "--output-path", str(dataset_path)],
    )

    module.main()

    output = capsys.readouterr().out
    assert "sample_size=24" in output
    assert "hallucination_count=10" in output
    assert "non_hallucination_count=14" in output
    assert "source_count=2" in output
    assert "hallucination_bucket_coverage=" in output
    assert "difficulty_heuristics=" in output
    assert "warnings=long-response coverage is too low" in output
    assert "artifact=" in output


def test_train_text_detector_script_prints_training_summary(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module("train_text_detector.py")
    dataset_path = tmp_path / "dataset.jsonl"
    features_path = tmp_path / "features.json"
    model_path = tmp_path / "detector.json"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "preprocess_textual_training_dataset",
        lambda **_: {
            "rows": [{"label": 0}, {"label": 1}],
            "summary": {"sample_size": 2, "feature_names": ["response_length"]},
            "artifact_path": str(features_path),
        },
    )
    monkeypatch.setattr(
        module,
        "train_detector_from_preprocessed_rows",
        lambda **_: {
            "train_size": 2,
            "dev_size": 1,
            "model_artifact_path": str(model_path),
            "dev_pr_auc": 0.71,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_text_detector.py", "--dataset-path", str(dataset_path), "--model-output-path", str(model_path)],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "train_size=2" in output
    assert "dev_pr_auc=0.7100" in output
    assert "model_artifact=" in output


def test_score_private_dataset_script_prints_output_path(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module("score_private_dataset.py")
    input_path = tmp_path / "private.csv"
    model_path = tmp_path / "detector.json"
    output_path = tmp_path / "scores.csv"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "score_private_dataset",
        lambda **_: {"sample_size": 3, "output_path": str(output_path)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_private_dataset.py",
            "--input-path",
            str(input_path),
            "--model-artifact-path",
            str(model_path),
            "--output-path",
            str(output_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "sample_size=3" in output
    assert "output=" in output
