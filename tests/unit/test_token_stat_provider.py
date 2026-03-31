from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from features.extractor import StructuralFeatureExtractor
from features.extractor import InternalModelSignal
from inference.token_stats import (
    CollectedModelSignals,
    GigaChatProviderConfig,
    GigaChatTokenStatProvider,
    TransformersProviderConfig,
    TransformersTokenStatProvider,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.bos_token_id = 0
        self._token_to_id = {
            "<bos>": 0,
            "hello": 1,
            "prompt": 2,
            "world": 3,
            "again": 4,
            "2024": 5,
        }
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}

    def __call__(self, text: str, add_special_tokens: bool = False):
        tokens = text.split() if text else []
        return {"input_ids": [self._token_to_id[token] for token in tokens]}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]


class FakeModelOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.to_calls = 0

    def to(self, device: str):
        self.to_calls += 1
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> FakeModelOutput:
        sequence = input_ids[0].tolist()
        vocab_size = max(sequence) + 2
        logits = torch.zeros((1, len(sequence), vocab_size), dtype=torch.float32)
        for index in range(len(sequence) - 1):
            next_token_id = sequence[index + 1]
            logits[0, index, next_token_id] = 5.0
            logits[0, index, (next_token_id + 1) % vocab_size] = 4.0
        hidden_states = None
        if output_hidden_states:
            hidden_size = 3
            layer_0 = torch.zeros((1, len(sequence), hidden_size), dtype=torch.float32)
            layer_1 = torch.tensor(
                [
                    [
                        [0.10, 0.00, 0.00],
                        [0.20, 0.00, 0.00],
                        [0.50, 0.20, 0.00],
                        [0.60, 0.10, 0.00],
                    ]
                ],
                dtype=torch.float32,
            )
            layer_2 = torch.tensor(
                [
                    [
                        [0.00, 0.10, 0.00],
                        [0.00, 0.20, 0.00],
                        [0.40, 0.30, 0.10],
                        [0.55, 0.12, 0.08],
                    ]
                ],
                dtype=torch.float32,
            )
            layer_3 = torch.tensor(
                [
                    [
                        [0.00, 0.00, 0.10],
                        [0.00, 0.00, 0.20],
                        [0.60, 0.10, 0.05],
                        [0.70, 0.08, 0.05],
                    ]
                ],
                dtype=torch.float32,
            )
            hidden_states = (layer_0, layer_1, layer_2, layer_3)
        return FakeModelOutput(logits, hidden_states=hidden_states)


class OffsetAwareTokenizer:
    bos_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        if text == "Hello":
            payload = {"input_ids": [1]}
        elif text == "world":
            payload = {"input_ids": [2]}
        elif text == "Helloworld":
            payload = {"input_ids": [10, 11, 12]}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(0, 1), (1, 5), (5, 10)]
        elif text == "Hello\n\n### Response:\nworld":
            payload = {"input_ids": [10, 20, 21, 12]}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(0, 5), (5, 7), (7, 18), (18, 23)]
        else:
            payload = {"input_ids": []}
        return payload

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        token_map = {
            1: "Hello",
            2: "standalone_world",
            10: "H",
            11: "ello",
            12: "merged_world",
            20: "delimiter_a",
            21: "delimiter_b",
        }
        return [token_map[token_id] for token_id in ids]


class MalformedModel:
    def __call__(self, *, input_ids: torch.Tensor):
        return object()


class MissingHiddenStatesModel(FakeModel):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> FakeModelOutput:
        return super().__call__(input_ids=input_ids, output_hidden_states=False)


class FakeDispatchedModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.hf_device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def to(self, device: str):
        raise AssertionError("dispatched model must not be moved with .to()")


class DelimiterAwareTokenizer:
    bos_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        payload_map = {
            "Hello|<resp>|world": {
                "input_ids": [10, 20, 30],
                "offset_mapping": [(0, 5), (5, 13), (13, 18)],
            },
            "|<resp>|world": {
                "input_ids": [20, 30],
                "offset_mapping": [(0, 8), (8, 13)],
            },
            "Hello|<resp>|": {
                "input_ids": [10, 20],
                "offset_mapping": [(0, 5), (5, 13)],
            },
            "Long prompt text|<resp>|long response text": {
                "input_ids": [40, 20, 50, 51, 52],
                "offset_mapping": [(0, 16), (16, 24), (24, 28), (29, 37), (38, 42)],
            },
        }
        payload = payload_map.get(text, {"input_ids": [], "offset_mapping": []})
        if not return_offsets_mapping:
            payload = {"input_ids": payload["input_ids"]}
        return payload

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        token_map = {
            10: "Hello",
            20: "<resp>",
            30: "world",
            40: "long_prompt",
            50: "long",
            51: "response",
            52: "text",
        }
        return [token_map[token_id] for token_id in ids]


def test_provider_returns_token_stat_output_structure() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="hello prompt", response="world again")

    assert len(token_stats) == 2
    assert token_stats[0].token == "world"
    assert isinstance(token_stats[0].logprob, float)
    assert isinstance(token_stats[0].entropy, float)
    assert isinstance(token_stats[0].top1_top2_margin, float)


def test_transformers_provider_returns_token_stat_output_structure() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="hello prompt", response="world again")

    assert len(token_stats) == 2
    assert token_stats[0].token == "world"
    assert isinstance(token_stats[0].logprob, float)
    assert isinstance(token_stats[0].entropy, float)
    assert isinstance(token_stats[0].top1_top2_margin, float)


def test_provider_is_deterministic_and_response_aligned() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    first = provider.collect(prompt="hello prompt", response="world again")
    second = provider.collect(prompt="hello prompt", response="world again")

    assert [stat.token for stat in first] == ["world", "again"]
    assert first == second


def test_transformers_provider_uses_combined_offsets_for_response_alignment() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=OffsetAwareTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="Hello", response="world")

    assert [stat.token for stat in token_stats] == ["merged_world"]


def test_provider_returns_finite_values() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="hello prompt", response="world 2024")

    assert all(torch.isfinite(torch.tensor(stat.logprob)) for stat in token_stats)
    assert all(torch.isfinite(torch.tensor(stat.entropy)) for stat in token_stats)
    assert all(
        torch.isfinite(torch.tensor(stat.top1_top2_margin)) for stat in token_stats
    )


def test_provider_handles_empty_and_short_response() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    assert provider.collect(prompt="hello prompt", response="") == []
    assert len(provider.collect(prompt="hello prompt", response="world")) == 1


def test_provider_uses_explicit_delimiter_for_boundary_alignment() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            response_delimiter="|<resp>|",
        ),
        tokenizer=DelimiterAwareTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="Hello", response="world")

    assert [stat.token for stat in token_stats] == ["world"]


def test_provider_handles_empty_prompt_with_explicit_delimiter() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            response_delimiter="|<resp>|",
        ),
        tokenizer=DelimiterAwareTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt="", response="world")

    assert [stat.token for stat in token_stats] == ["world"]


def test_provider_handles_empty_response_with_explicit_delimiter() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            response_delimiter="|<resp>|",
        ),
        tokenizer=DelimiterAwareTokenizer(),
        model=FakeModel(),
    )

    assert provider.collect(prompt="Hello", response="") == []


@pytest.mark.parametrize(
    ("prompt", "response", "expected_tokens"),
    [
        ("Hello", "world", ["world"]),
        ("Long prompt text", "long response text", ["long", "response", "text"]),
    ],
)
def test_provider_boundary_is_stable_across_short_and_long_examples(
    prompt: str,
    response: str,
    expected_tokens: list[str],
) -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            response_delimiter="|<resp>|",
        ),
        tokenizer=DelimiterAwareTokenizer(),
        model=FakeModel(),
    )

    token_stats = provider.collect(prompt=prompt, response=response)

    assert [stat.token for stat in token_stats] == expected_tokens


def test_provider_output_is_compatible_with_existing_extractor() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    token_stats = provider.collect(prompt="hello prompt", response="world again")
    features = extractor.extract(
        prompt="hello prompt",
        response="world again",
        token_stats=token_stats,
    )

    assert "token_mean_logprob" in features
    assert "token_entropy_mean" in features
    assert "token_top1_top2_margin_mean" in features


def test_transformers_provider_output_is_compatible_with_existing_extractor() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    token_stats = provider.collect(prompt="hello prompt", response="world again")
    features = extractor.extract(
        prompt="hello prompt",
        response="world again",
        token_stats=token_stats,
    )

    assert "token_mean_logprob" in features
    assert "token_entropy_mean" in features
    assert "token_top1_top2_margin_mean" in features


def test_provider_collect_signals_returns_internal_signal_when_enabled() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    collected = provider.collect_signals(prompt="hello prompt", response="world again")

    assert isinstance(collected, CollectedModelSignals)
    assert len(collected.token_stats) == 2
    assert collected.internal_signal is not None
    assert collected.internal_signal.last_layer_pooled_l2 > 0.0
    assert collected.internal_signal.selected_layer_norm_variance >= 0.0
    assert collected.internal_signal.layer_disagreement_mean >= 0.0


def test_provider_collect_signals_is_deterministic_and_finite() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    first = provider.collect_signals(prompt="hello prompt", response="world again")
    second = provider.collect_signals(prompt="hello prompt", response="world again")

    assert first == second
    assert first.internal_signal is not None
    assert torch.isfinite(torch.tensor(first.internal_signal.last_layer_pooled_l2))
    assert torch.isfinite(
        torch.tensor(first.internal_signal.selected_layer_norm_variance)
    )
    assert torch.isfinite(torch.tensor(first.internal_signal.layer_disagreement_mean))


def test_provider_fails_clearly_when_internal_features_are_enabled_but_hidden_states_are_missing() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
        tokenizer=FakeTokenizer(),
        model=MissingHiddenStatesModel(),
    )

    with pytest.raises(ValueError, match="hidden states"):
        provider.collect_signals(prompt="hello prompt", response="world again")


def test_provider_fails_clearly_on_malformed_model_output() -> None:
    provider = GigaChatTokenStatProvider(
        config=GigaChatProviderConfig(model_id="ai-sage/GigaChat3-10B-A1.8B-bf16"),
        tokenizer=FakeTokenizer(),
        model=MalformedModel(),
    )

    with pytest.raises(ValueError, match="logits"):
        provider.collect(prompt="hello prompt", response="world again")


def test_provider_fails_clearly_when_checkpoint_cannot_be_loaded(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "GigaChat3"
    checkpoint_dir.mkdir()

    def _raise_load_error(*args, **kwargs):
        raise OSError("failed to load checkpoint")

    monkeypatch.setattr(
        "inference.token_stats.AutoTokenizer.from_pretrained",
        _raise_load_error,
    )

    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="ai-sage/GigaChat3-10B-A1.8B-bf16",
            checkpoint_path=str(checkpoint_dir),
        ),
    )

    with pytest.raises(RuntimeError, match="failed to load token-stat provider"):
        provider.collect(prompt="hello prompt", response="world again")


def test_provider_loads_single_device_model_without_device_map(monkeypatch) -> None:
    fake_model = FakeModel()
    load_kwargs = {}

    def _load_model(*args, **kwargs):
        load_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        "inference.token_stats.AutoModelForCausalLM.from_pretrained",
        _load_model,
    )

    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2", device="cpu"),
    )

    loaded_model = provider._get_model()

    assert loaded_model is fake_model
    assert "device_map" not in load_kwargs
    assert fake_model.device == "cpu"
    assert fake_model.to_calls == 1


def test_provider_loads_auto_device_map_without_redundant_to(monkeypatch) -> None:
    fake_model = FakeDispatchedModel()
    load_kwargs = {}

    def _load_model(*args, **kwargs):
        load_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        "inference.token_stats.AutoModelForCausalLM.from_pretrained",
        _load_model,
    )

    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="ai-sage/GigaChat3-10B-A1.8B-bf16",
            device="auto",
            max_memory={0: "14GiB", 1: "14GiB"},
        ),
    )

    loaded_model = provider._get_model()

    assert loaded_model is fake_model
    assert load_kwargs["device_map"] == "auto"
    assert load_kwargs["max_memory"] == {0: "14GiB", 1: "14GiB"}


def test_dispatched_model_uses_cpu_input_path() -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2", device="auto"),
        tokenizer=FakeTokenizer(),
        model=FakeDispatchedModel(),
    )

    input_device = provider._input_device_for_model(provider._get_model())

    assert input_device == "cpu"


def test_config_normalizes_string_integer_max_memory_keys_from_json(tmp_path) -> None:
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        '{'
        '"model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16", '
        '"max_memory": {"0": "14GiB", "1": "14GiB"}'
        '}'
    )

    config = TransformersProviderConfig.from_json(config_path)

    assert config.max_memory == {0: "14GiB", 1: "14GiB"}


def test_config_keeps_integer_max_memory_keys() -> None:
    config = TransformersProviderConfig(
        model_id="ai-sage/GigaChat3-10B-A1.8B-bf16",
        max_memory={0: "14GiB", 1: "14GiB"},
    )

    assert config.max_memory == {0: "14GiB", 1: "14GiB"}


def test_config_rejects_cuda_prefixed_max_memory_keys(tmp_path) -> None:
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        '{'
        '"model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16", '
        '"max_memory": {"cuda:0": "14GiB", "cuda:1": "14GiB"}'
        '}'
    )

    with pytest.raises(ValueError, match="max_memory keys must be integer GPU ids"):
        TransformersProviderConfig.from_json(config_path)
