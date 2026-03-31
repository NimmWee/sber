from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.default_detector import build_default_detector_extractor
from eval.runner import RawExampleEvaluationDataset
from inference.token_stats import TransformersTokenStatProvider
from models.head import train_logistic_regression_head
from utils.latency import (
    LatencyBenchmarkConfig,
    benchmark_single_example_latency,
    benchmark_single_example_latency_with_provider,
)
from utils.script_helpers import (
    build_smoke_examples,
    resolve_transformers_provider_config,
    write_json_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "latency_real_provider"),
    )
    parser.add_argument("--baseline-repeat-count", type=int, default=10)
    parser.add_argument("--provider-repeat-count", type=int, default=5)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_examples, validation_examples = build_smoke_examples()
    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)

    baseline_dataset = RawExampleEvaluationDataset(
        train_examples=train_examples,
        validation_examples=validation_examples,
        extractor=build_default_detector_extractor(),
    )
    provider_dataset = RawExampleEvaluationDataset(
        train_examples=train_examples,
        validation_examples=validation_examples,
        extractor=build_default_detector_extractor(),
        token_stat_provider=provider,
    )

    baseline_split = baseline_dataset.load_split()
    provider_split = provider_dataset.load_split()
    baseline_head = train_logistic_regression_head(
        baseline_split.train_features,
        baseline_split.train_labels,
    )
    provider_head = train_logistic_regression_head(
        provider_split.train_features,
        provider_split.train_labels,
    )

    probe_example = validation_examples[-1]
    baseline_result = benchmark_single_example_latency(
        prompt=probe_example.prompt,
        response=probe_example.response,
        extractor=build_default_detector_extractor(),
        head=baseline_head,
        config=LatencyBenchmarkConfig(
            repeat_count=args.baseline_repeat_count,
            artifact_dir=artifact_dir / "baseline",
        ),
    )
    provider_result = benchmark_single_example_latency_with_provider(
        prompt=probe_example.prompt,
        response=probe_example.response,
        extractor=build_default_detector_extractor(),
        head=provider_head,
        token_stat_provider=provider,
        config=LatencyBenchmarkConfig(
            repeat_count=args.provider_repeat_count,
            artifact_dir=artifact_dir / "provider",
        ),
    )

    payload = {
        "model_source": config.model_source,
        "response_delimiter": config.response_delimiter,
        "baseline_total_p50_ms": baseline_result.total.p50_ms,
        "baseline_total_p95_ms": baseline_result.total.p95_ms,
        "provider_total_p50_ms": provider_result.total.p50_ms,
        "provider_total_p95_ms": provider_result.total.p95_ms,
        "token_stat_collection_p50_ms": provider_result.token_stat_collection.p50_ms,
        "feature_aggregation_p50_ms": provider_result.feature_aggregation.p50_ms,
        "head_inference_p50_ms": provider_result.head_inference.p50_ms,
        "default_detector": "structural + base token uncertainty only",
    }
    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="latency_real_provider_summary.json",
        payload=payload,
    )

    print(f"baseline_total_p50_ms={baseline_result.total.p50_ms:.4f}")
    print(f"provider_total_p50_ms={provider_result.total.p50_ms:.4f}")
    print(
        "provider_breakdown_p50_ms="
        f"{provider_result.token_stat_collection.p50_ms:.4f}/"
        f"{provider_result.feature_aggregation.p50_ms:.4f}/"
        f"{provider_result.head_inference.p50_ms:.4f}"
    )
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
