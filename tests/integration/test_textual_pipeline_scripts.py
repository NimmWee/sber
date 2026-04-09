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
