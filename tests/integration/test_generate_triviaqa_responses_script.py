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


def test_generate_triviaqa_responses_sets_pad_token_from_eos_for_causal_lm() -> None:
    module = _load_script_module("generate_triviaqa_responses.py")

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "</s>"
            self.pad_token_id = None
            self.eos_token_id = 7

        def __call__(self, prompts, **kwargs):
            assert self.pad_token == self.eos_token
            assert kwargs["padding"] is True
            assert kwargs["return_tensors"] == "pt"
            return {
                "input_ids": module.torch.tensor([[1, 2], [1, 2]]),
                "attention_mask": module.torch.tensor([[1, 1], [1, 1]]),
            }

        def decode(self, generated_tokens, skip_special_tokens=True):
            return "generated"

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["pad_token_id"] == 7
            return module.torch.tensor([[1, 2, 3], [1, 2, 4]])

    class FakeProvider:
        def __init__(self) -> None:
            self.config = type("Config", (), {"response_delimiter": "\n\n### Response:\n"})()
            self._tokenizer_instance = FakeTokenizer()

        def _get_tokenizer(self):
            return self._tokenizer_instance

        def _get_model(self):
            return FakeModel()

        def _input_device_for_model(self, model):
            return "cpu"

    examples = [
        type("Example", (), {"prompt": "Q1"})(),
        type("Example", (), {"prompt": "Q2"})(),
    ]

    responses = module._generate_responses_for_examples(
        examples=examples,
        token_stat_provider=FakeProvider(),
        batch_size=2,
    )

    assert responses == ["generated", "generated"]


def test_generate_triviaqa_responses_batch_size_one_returns_plain_strings() -> None:
    module = _load_script_module("generate_triviaqa_responses.py")

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = "<pad>"
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.eos_token_id = 7

        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": module.torch.tensor([[10, 11]]),
                "attention_mask": module.torch.tensor([[1, 1]]),
            }

        def decode(self, generated_tokens, skip_special_tokens=True):
            return "single-response"

    class FakeModel:
        def generate(self, **kwargs):
            return module.torch.tensor([[10, 11, 12]])

    class FakeProvider:
        def __init__(self) -> None:
            self.config = type("Config", (), {"response_delimiter": "\n\n### Response:\n"})()
            self._tokenizer_instance = FakeTokenizer()

        def _get_tokenizer(self):
            return self._tokenizer_instance

        def _get_model(self):
            return FakeModel()

        def _input_device_for_model(self, model):
            return "cpu"

    responses = module._generate_responses_for_examples(
        examples=[type("Example", (), {"prompt": "Q1"})()],
        token_stat_provider=FakeProvider(),
        batch_size=1,
    )

    assert responses == ["single-response"]
    assert isinstance(responses[0], str)


def test_generate_triviaqa_responses_batch_size_greater_than_one_decodes_all_outputs() -> None:
    module = _load_script_module("generate_triviaqa_responses.py")

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = "<pad>"
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.eos_token_id = 7

        def __call__(self, prompts, **kwargs):
            assert len(prompts) == 2
            return {
                "input_ids": module.torch.tensor([[10, 11], [20, 21]]),
                "attention_mask": module.torch.tensor([[1, 1], [1, 1]]),
            }

        def decode(self, generated_tokens, skip_special_tokens=True):
            token_list = generated_tokens.tolist()
            if token_list == [12]:
                return "first-batch-response"
            if token_list == [22]:
                return "second-batch-response"
            raise AssertionError(f"unexpected generated tokens: {token_list}")

    class FakeModel:
        def generate(self, **kwargs):
            return module.torch.tensor([[10, 11, 12], [20, 21, 22]])

    class FakeProvider:
        def __init__(self) -> None:
            self.config = type("Config", (), {"response_delimiter": "\n\n### Response:\n"})()
            self._tokenizer_instance = FakeTokenizer()

        def _get_tokenizer(self):
            return self._tokenizer_instance

        def _get_model(self):
            return FakeModel()

        def _input_device_for_model(self, model):
            return "cpu"

    responses = module._generate_responses_for_examples(
        examples=[
            type("Example", (), {"prompt": "Q1"})(),
            type("Example", (), {"prompt": "Q2"})(),
        ],
        token_stat_provider=FakeProvider(),
        batch_size=2,
    )

    assert responses == ["first-batch-response", "second-batch-response"]


def test_generate_triviaqa_responses_batch_roundtrip_writes_valid_jsonl(tmp_path) -> None:
    module = _load_script_module("generate_triviaqa_responses.py")
    dataset_path = tmp_path / "triviaqa.jsonl"
    output_path = tmp_path / "generated.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q1", "answer": "A1"}),
                json.dumps({"question": "Q2", "answer": "A2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeProvider:
        config = type("Config", (), {"response_delimiter": "\n\n### Response:\n"})()

    module._generate_responses_for_examples = lambda **_: [
        'first "quoted" answer\nline two',
        "Ответ 2",
    ]

    summary = module.generate_triviaqa_responses(
        dataset_path=dataset_path,
        output_path=output_path,
        token_stat_provider=FakeProvider(),
        max_samples=2,
        batch_size=2,
    )

    assert summary["processed_samples"] == 2
    written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written_rows[0]["response"] == 'first "quoted" answer\nline two'
    assert written_rows[1]["response"] == "Ответ 2"
