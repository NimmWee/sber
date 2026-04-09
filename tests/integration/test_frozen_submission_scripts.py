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


def test_train_frozen_submission_script_prints_summary(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module("train_frozen_submission.py")
    dataset_path = tmp_path / "dataset.jsonl"
    artifact_dir = tmp_path / "frozen_best"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "train_frozen_best_submission",
        lambda **_: {
            "train_size": 10,
            "dev_size": 4,
            "dev_pr_auc": 0.55,
            "artifact_dir": str(artifact_dir),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_frozen_submission.py",
            "--dataset-path",
            str(dataset_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "train_size=10" in output
    assert "dev_pr_auc=0.5500" in output
    assert "artifact_dir=" in output


def test_score_frozen_submission_script_prints_output(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module("score_frozen_submission.py")
    input_path = tmp_path / "private.csv"
    output_path = tmp_path / "scores.csv"
    artifact_dir = tmp_path / "frozen_best"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "score_private_frozen_submission",
        lambda **_: {
            "sample_size": 3,
            "output_path": str(output_path),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_frozen_submission.py",
            "--input-path",
            str(input_path),
            "--artifact-dir",
            str(artifact_dir),
            "--output-path",
            str(output_path),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "sample_size=3" in output
    assert "output=" in output


def test_train_frozen_submission_script_fails_with_actionable_message_for_missing_dataset(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_script_module("train_frozen_submission.py")
    dataset_path = tmp_path / "missing.jsonl"
    artifact_dir = tmp_path / "frozen_best"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_frozen_submission.py",
            "--dataset-path",
            str(dataset_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    try:
        module.main()
    except FileNotFoundError as error:
        assert "textual training dataset was not found" in str(error)
        assert "build_text_training_dataset.py" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError for missing training dataset")


def test_score_frozen_submission_script_fails_with_actionable_message_for_missing_private_csv(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_script_module("score_frozen_submission.py")
    input_path = tmp_path / "missing_private.csv"
    artifact_dir = tmp_path / "frozen_best"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_frozen_submission.py",
            "--input-path",
            str(input_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    try:
        module.main()
    except FileNotFoundError as error:
        assert "knowledge_bench_private.csv was not found" in str(error)
        assert "data/bench/knowledge_bench_private.csv" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError for missing private benchmark input")
