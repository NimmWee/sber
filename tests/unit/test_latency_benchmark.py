from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from features.extractor import StructuralFeatureExtractor
from inference.token_stats import TransformersProviderConfig, TransformersTokenStatProvider
from models.head import train_logistic_regression_head
from utils.latency import (
    LatencyBenchmarkConfig,
    LatencyBenchmarkResult,
    benchmark_single_example_latency,
    benchmark_single_example_latency_with_provider,
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
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    def __call__(self, *, input_ids: torch.Tensor) -> FakeModelOutput:
        sequence = input_ids[0].tolist()
        vocab_size = 6
        logits = torch.zeros((1, len(sequence), vocab_size), dtype=torch.float32)
        for index in range(len(sequence) - 1):
            next_token_id = sequence[index + 1]
            logits[0, index, next_token_id] = 5.0
            logits[0, index, (next_token_id + 1) % vocab_size] = 4.0
        return FakeModelOutput(logits)


def _train_small_head():
    feature_rows = [
        {"response_length": 10.0, "digit_count": 0.0, "novelty_ratio_proxy": 0.1},
        {"response_length": 20.0, "digit_count": 0.0, "novelty_ratio_proxy": 0.2},
        {"response_length": 80.0, "digit_count": 2.0, "novelty_ratio_proxy": 0.8},
        {"response_length": 90.0, "digit_count": 3.0, "novelty_ratio_proxy": 0.9},
    ]
    labels = [0, 0, 1, 1]
    return train_logistic_regression_head(feature_rows, labels)


def test_benchmark_validates_single_example_input() -> None:
    extractor = StructuralFeatureExtractor()
    head = _train_small_head()
    config = LatencyBenchmarkConfig(repeat_count=3)

    with pytest.raises(TypeError, match="prompt and response must be strings"):
        benchmark_single_example_latency(  # type: ignore[arg-type]
            prompt=None,
            response="Answer",
            extractor=extractor,
            head=head,
            config=config,
        )


def test_benchmark_returns_structured_result(tmp_path) -> None:
    extractor = StructuralFeatureExtractor()
    head = _train_small_head()
    config = LatencyBenchmarkConfig(repeat_count=3, artifact_dir=tmp_path)

    result = benchmark_single_example_latency(
        prompt="What is the capital of France?",
        response="Paris is the capital of France.",
        extractor=extractor,
        head=head,
        config=config,
    )

    assert isinstance(result, LatencyBenchmarkResult)


def test_benchmark_includes_feature_head_and_total_timings(tmp_path) -> None:
    extractor = StructuralFeatureExtractor()
    head = _train_small_head()
    config = LatencyBenchmarkConfig(repeat_count=5, artifact_dir=tmp_path)

    result = benchmark_single_example_latency(
        prompt="What year was the treaty signed?",
        response="The treaty was signed in 2024.",
        extractor=extractor,
        head=head,
        config=config,
    )

    assert result.feature_extraction.p50_ms >= 0.0
    assert result.head_inference.p50_ms >= 0.0
    assert result.total.p50_ms >= 0.0


def test_benchmark_creates_summary_artifact(tmp_path) -> None:
    extractor = StructuralFeatureExtractor()
    head = _train_small_head()
    config = LatencyBenchmarkConfig(repeat_count=3, artifact_dir=tmp_path)

    result = benchmark_single_example_latency(
        prompt="How many samples returned?",
        response="The mission returned 3 samples.",
        extractor=extractor,
        head=head,
        config=config,
    )

    assert result.artifact_path is not None
    assert Path(result.artifact_path).exists()


def test_invalid_benchmark_configuration_fails_clearly() -> None:
    with pytest.raises(ValueError, match="repeat_count must be positive"):
        LatencyBenchmarkConfig(repeat_count=0)


def test_provider_backed_benchmark_separates_provider_feature_head_and_total(
    tmp_path,
) -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)
    head = _train_small_head()
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    config = LatencyBenchmarkConfig(repeat_count=5, artifact_dir=tmp_path)

    result = benchmark_single_example_latency_with_provider(
        prompt="hello prompt",
        response="world again",
        extractor=extractor,
        head=head,
        token_stat_provider=provider,
        config=config,
    )

    assert result.token_stat_collection.p50_ms >= 0.0
    assert result.feature_aggregation.p50_ms >= 0.0
    assert result.head_inference.p50_ms >= 0.0
    assert result.total.p50_ms >= 0.0
