import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from features.extractor import FeatureExtractor
from inference.token_stats import TokenStatProvider
from models.head import TrainedLogisticRegressionHead


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float
    p95_ms: float


class LatencyRunner(Protocol):
    def run(self) -> LatencySummary:
        ...


@dataclass(frozen=True)
class LatencyBenchmarkConfig:
    repeat_count: int
    artifact_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.repeat_count <= 0:
            raise ValueError("repeat_count must be positive")


@dataclass(frozen=True)
class LatencyPhaseSummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    repeat_count: int
    feature_extraction: LatencyPhaseSummary
    head_inference: LatencyPhaseSummary
    total: LatencyPhaseSummary
    artifact_path: str | None = None


@dataclass(frozen=True)
class ProviderBackedLatencyBenchmarkResult:
    repeat_count: int
    token_stat_collection: LatencyPhaseSummary
    feature_aggregation: LatencyPhaseSummary
    head_inference: LatencyPhaseSummary
    total: LatencyPhaseSummary
    artifact_path: str | None = None


def benchmark_single_example_latency(
    *,
    prompt: str,
    response: str,
    extractor: FeatureExtractor,
    head: TrainedLogisticRegressionHead,
    config: LatencyBenchmarkConfig,
) -> LatencyBenchmarkResult:
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must be strings")

    feature_extraction_samples_ms: list[float] = []
    head_inference_samples_ms: list[float] = []
    total_samples_ms: list[float] = []

    feature_row = extractor.extract(prompt=prompt, response=response)
    for _ in range(config.repeat_count):
        extraction_start = time.perf_counter()
        extractor.extract(prompt=prompt, response=response)
        extraction_elapsed_ms = (time.perf_counter() - extraction_start) * 1000.0
        feature_extraction_samples_ms.append(extraction_elapsed_ms)

        head_start = time.perf_counter()
        head.predict_proba(feature_row)
        head_elapsed_ms = (time.perf_counter() - head_start) * 1000.0
        head_inference_samples_ms.append(head_elapsed_ms)

        total_start = time.perf_counter()
        total_features = extractor.extract(prompt=prompt, response=response)
        head.predict_proba(total_features)
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        total_samples_ms.append(total_elapsed_ms)

    result = LatencyBenchmarkResult(
        repeat_count=config.repeat_count,
        feature_extraction=_summarize_samples(feature_extraction_samples_ms),
        head_inference=_summarize_samples(head_inference_samples_ms),
        total=_summarize_samples(total_samples_ms),
    )

    if config.artifact_dir is None:
        return result

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "latency_benchmark.json"
    artifact_path.write_text(json.dumps(asdict(result), indent=2))
    return LatencyBenchmarkResult(
        repeat_count=result.repeat_count,
        feature_extraction=result.feature_extraction,
        head_inference=result.head_inference,
        total=result.total,
        artifact_path=str(artifact_path),
    )


def benchmark_single_example_latency_with_provider(
    *,
    prompt: str,
    response: str,
    extractor: FeatureExtractor,
    head: TrainedLogisticRegressionHead,
    token_stat_provider: TokenStatProvider,
    config: LatencyBenchmarkConfig,
) -> ProviderBackedLatencyBenchmarkResult:
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must be strings")

    provider_samples_ms: list[float] = []
    feature_samples_ms: list[float] = []
    head_samples_ms: list[float] = []
    total_samples_ms: list[float] = []

    for _ in range(config.repeat_count):
        provider_start = time.perf_counter()
        token_stats = token_stat_provider.collect(prompt=prompt, response=response)
        provider_elapsed_ms = (time.perf_counter() - provider_start) * 1000.0
        provider_samples_ms.append(provider_elapsed_ms)

        feature_start = time.perf_counter()
        features = extractor.extract(
            prompt=prompt,
            response=response,
            token_stats=token_stats,
        )
        feature_elapsed_ms = (time.perf_counter() - feature_start) * 1000.0
        feature_samples_ms.append(feature_elapsed_ms)

        head_start = time.perf_counter()
        head.predict_proba(features)
        head_elapsed_ms = (time.perf_counter() - head_start) * 1000.0
        head_samples_ms.append(head_elapsed_ms)

        total_start = time.perf_counter()
        total_token_stats = token_stat_provider.collect(prompt=prompt, response=response)
        total_features = extractor.extract(
            prompt=prompt,
            response=response,
            token_stats=total_token_stats,
        )
        head.predict_proba(total_features)
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        total_samples_ms.append(total_elapsed_ms)

    result = ProviderBackedLatencyBenchmarkResult(
        repeat_count=config.repeat_count,
        token_stat_collection=_summarize_samples(provider_samples_ms),
        feature_aggregation=_summarize_samples(feature_samples_ms),
        head_inference=_summarize_samples(head_samples_ms),
        total=_summarize_samples(total_samples_ms),
    )
    if config.artifact_dir is None:
        return result

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "provider_latency_benchmark.json"
    artifact_path.write_text(json.dumps(asdict(result), indent=2))
    return ProviderBackedLatencyBenchmarkResult(
        repeat_count=result.repeat_count,
        token_stat_collection=result.token_stat_collection,
        feature_aggregation=result.feature_aggregation,
        head_inference=result.head_inference,
        total=result.total,
        artifact_path=str(artifact_path),
    )


def _summarize_samples(samples_ms: list[float]) -> LatencyPhaseSummary:
    sorted_samples = sorted(samples_ms)
    return LatencyPhaseSummary(
        mean_ms=statistics.fmean(sorted_samples),
        p50_ms=_percentile(sorted_samples, 0.5),
        p95_ms=_percentile(sorted_samples, 0.95),
    )


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]

    position = (len(sorted_samples) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_samples) - 1)
    fraction = position - lower_index
    lower_value = sorted_samples[lower_index]
    upper_value = sorted_samples[upper_index]
    return lower_value + (upper_value - lower_value) * fraction
