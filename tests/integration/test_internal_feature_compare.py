import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.internal_compare import compare_base_vs_internal_features
from eval.runner import RawLabeledExample
from inference.token_stats import TransformersProviderConfig, TransformersTokenStatProvider


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
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> FakeModelOutput:
        sequence = input_ids[0].tolist()
        vocab_size = 6
        logits = torch.zeros((1, len(sequence), vocab_size), dtype=torch.float32)
        for index in range(len(sequence) - 1):
            next_token_id = sequence[index + 1]
            logits[0, index, next_token_id] = 5.0
            logits[0, index, (next_token_id + 1) % vocab_size] = 4.0

        hidden_states = None
        if output_hidden_states:
            hidden_states = (
                torch.zeros((1, len(sequence), 3), dtype=torch.float32),
                torch.tensor(
                    [[[0.10, 0.00, 0.00], [0.20, 0.00, 0.00], [0.50, 0.20, 0.00], [0.60, 0.10, 0.00]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[0.00, 0.10, 0.00], [0.00, 0.20, 0.00], [0.40, 0.30, 0.10], [0.55, 0.12, 0.08]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[0.00, 0.00, 0.10], [0.00, 0.00, 0.20], [0.60, 0.10, 0.05], [0.70, 0.08, 0.05]]],
                    dtype=torch.float32,
                ),
            )
        return FakeModelOutput(logits, hidden_states=hidden_states)


class CountingProvider:
    def __init__(self, *, model_id: str, enable_internal_features: bool) -> None:
        self.config = TransformersProviderConfig(
            model_id=model_id,
            enable_internal_features=enable_internal_features,
            selected_hidden_layers=(-1, -2),
        )
        self.share_backend_calls = 0
        self.collect_calls = 0
        self.collect_signals_calls = 0
        self.shared_child_configs: list[TransformersProviderConfig] = []

    def collect(self, prompt: str, response: str):
        self.collect_calls += 1
        return self.collect_signals(prompt=prompt, response=response).token_stats

    def collect_signals(self, prompt: str, response: str):
        from inference.token_stats import CollectedModelSignals
        from features.extractor import TokenUncertaintyStat, InternalModelSignal

        self.collect_signals_calls += 1
        token_stats = [
            TokenUncertaintyStat("world", -0.1, 0.2, 0.6),
            TokenUncertaintyStat("again", -0.2, 0.3, 0.5),
        ]
        internal_signal = None
        if self.config.enable_internal_features:
            internal_signal = InternalModelSignal(
                last_layer_pooled_l2=1.2,
                last_layer_pooled_mean_abs=0.4,
                selected_layer_norm_variance=0.08,
                layer_disagreement_mean=0.12,
            )
        return CollectedModelSignals(
            token_stats=token_stats,
            internal_signal=internal_signal,
        )

    def share_backend(self, *, config: TransformersProviderConfig):
        self.share_backend_calls += 1
        self.shared_child_configs.append(config)
        child = CountingProvider(
            model_id=config.model_id,
            enable_internal_features=config.enable_internal_features,
        )
        child.collect_calls = self.collect_calls
        child.collect_signals_calls = self.collect_signals_calls
        return child


def test_compare_base_vs_internal_features_writes_summary_artifact(tmp_path) -> None:
    examples_train = [
        RawLabeledExample(prompt="hello prompt", response="world again", label=0),
        RawLabeledExample(prompt="hello prompt", response="world 2024", label=1),
        RawLabeledExample(prompt="hello prompt", response="world again", label=0),
        RawLabeledExample(prompt="hello prompt", response="world 2024", label=1),
    ]
    examples_validation = [
        RawLabeledExample(prompt="hello prompt", response="world again", label=0),
        RawLabeledExample(prompt="hello prompt", response="world 2024", label=1),
    ]
    base_provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    internal_provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    summary = compare_base_vs_internal_features(
        train_examples=examples_train,
        validation_examples=examples_validation,
        base_provider=base_provider,
        internal_provider=internal_provider,
        artifact_dir=tmp_path,
        latency_repeat_count=3,
    )

    assert 0.0 <= summary["baseline"]["pr_auc"] <= 1.0
    assert 0.0 <= summary["internal"]["pr_auc"] <= 1.0
    assert "pr_auc_delta" in summary
    assert "latency_delta_ms" in summary
    assert "recommendation" in summary
    assert Path(summary["artifact_path"]).exists()

    payload = json.loads(Path(summary["artifact_path"]).read_text())
    assert payload["baseline"]["model_artifact_path"]
    assert payload["internal"]["model_artifact_path"]


def test_compare_base_vs_internal_features_reuses_shared_provider_backend(tmp_path) -> None:
    examples_train = [
        RawLabeledExample(prompt="hello prompt", response="world again", label=0),
        RawLabeledExample(prompt="hello prompt", response="world 2024", label=1),
    ]
    examples_validation = [
        RawLabeledExample(prompt="hello prompt", response="world again", label=0),
        RawLabeledExample(prompt="hello prompt", response="world 2024", label=1),
    ]
    root_provider = CountingProvider(
        model_id="distilgpt2",
        enable_internal_features=True,
    )

    compare_base_vs_internal_features(
        train_examples=examples_train,
        validation_examples=examples_validation,
        base_provider=None,
        internal_provider=root_provider,
        artifact_dir=tmp_path,
        latency_repeat_count=1,
    )

    assert root_provider.share_backend_calls == 1
    assert root_provider.shared_child_configs[0].enable_internal_features is False
