from __future__ import annotations

import importlib.util
import json
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


def test_generate_triviaqa_responses_script_writes_expected_schema(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    module = _load_script_module("generate_triviaqa_responses.py")
    dataset_path = tmp_path / "triviaqa.jsonl"
    output_path = tmp_path / "generated.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Who wrote Hamlet?", "answer": "William Shakespeare"}),
                json.dumps({"question": "What is the capital of France?", "answer": "Paris"}),
            ]
        ),
        encoding="utf-8",
    )

    class FakeConfig:
        model_source = "/kaggle/temp/GigaChat3"
        response_delimiter = "\n\n### Response:\n"

    class FakeProvider:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(module, "resolve_transformers_provider_config", lambda **_: FakeConfig())
    monkeypatch.setattr(module, "resolve_triviaqa_path", lambda **_: dataset_path)
    monkeypatch.setattr(module, "TransformersTokenStatProvider", FakeProvider)
    monkeypatch.setattr(
        module,
        "generate_triviaqa_responses",
        lambda **_: {
            "processed_samples": 2,
            "output_path": str(output_path),
            "average_response_length": 13.5,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_triviaqa_responses.py",
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "model=/kaggle/temp/GigaChat3" in output
    assert "processed_samples=2" in output
    assert "output_path=" in output
    assert "average_response_length=13.50" in output
