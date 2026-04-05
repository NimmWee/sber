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


def test_label_triviaqa_generated_responses_script_prints_summary(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    module = _load_script_module("label_triviaqa_generated_responses.py")
    input_path = tmp_path / "generated.jsonl"
    output_path = tmp_path / "labeled.jsonl"

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"
        response_delimiter = "\n\n### Response:\n"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "label_triviaqa_generated_responses",
        lambda **_: {
            "total_labeled_samples": 8,
            "hallucination_count": 3,
            "non_hallucination_count": 5,
            "hallucination_ratio": 0.375,
            "output_path": str(output_path),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_triviaqa_generated_responses.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--batch-size",
            "4",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "total_labeled_samples=8" in output
    assert "hallucination_count=3" in output
    assert "non_hallucination_count=5" in output
    assert "hallucination_ratio=0.3750" in output
    assert "output_path=" in output
