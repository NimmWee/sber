import json
from pathlib import Path

from eval.ablation import recommend_feature_group
from eval.runner import RawExampleEvaluationDataset, RawLabeledExample, TrainValidationEvaluationRunner
from features.extractor import StructuralFeatureExtractor
from inference.token_stats import TokenStatProvider
from models.head import TrainedLogisticRegressionHead
from utils.latency import LatencyBenchmarkConfig, benchmark_single_example_latency_with_provider


def compare_base_vs_internal_features(
    *,
    train_examples: list[RawLabeledExample],
    validation_examples: list[RawLabeledExample],
    base_provider: TokenStatProvider,
    internal_provider: TokenStatProvider,
    artifact_dir: str | Path,
    latency_repeat_count: int = 20,
) -> dict:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_runner = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
            token_stat_provider=base_provider,
        ),
        artifact_dir=output_dir / "baseline",
    )
    baseline_summary = baseline_runner.run()
    baseline_head = TrainedLogisticRegressionHead.load(baseline_summary.model_artifact_path)

    internal_runner = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=StructuralFeatureExtractor(
                enable_token_uncertainty=True,
                enable_internal_features=True,
            ),
            token_stat_provider=internal_provider,
        ),
        artifact_dir=output_dir / "internal",
    )
    internal_summary = internal_runner.run()
    internal_head = TrainedLogisticRegressionHead.load(internal_summary.model_artifact_path)

    latency_example = validation_examples[0]
    baseline_latency = benchmark_single_example_latency_with_provider(
        prompt=latency_example.prompt,
        response=latency_example.response,
        extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
        head=baseline_head,
        token_stat_provider=base_provider,
        config=LatencyBenchmarkConfig(
            repeat_count=latency_repeat_count,
            artifact_dir=output_dir / "baseline_latency",
        ),
    )
    internal_latency = benchmark_single_example_latency_with_provider(
        prompt=latency_example.prompt,
        response=latency_example.response,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            enable_internal_features=True,
        ),
        head=internal_head,
        token_stat_provider=internal_provider,
        config=LatencyBenchmarkConfig(
            repeat_count=latency_repeat_count,
            artifact_dir=output_dir / "internal_latency",
        ),
    )

    pr_auc_delta = internal_summary.pr_auc - baseline_summary.pr_auc
    latency_delta_ms = internal_latency.total.mean_ms - baseline_latency.total.mean_ms
    payload = {
        "baseline": {
            "pr_auc": baseline_summary.pr_auc,
            "sample_size": baseline_summary.sample_size,
            "model_artifact_path": baseline_summary.model_artifact_path,
            "summary_artifact_path": baseline_summary.summary_artifact_path,
            "latency_mean_ms": baseline_latency.total.mean_ms,
            "latency_p95_ms": baseline_latency.total.p95_ms,
        },
        "internal": {
            "pr_auc": internal_summary.pr_auc,
            "sample_size": internal_summary.sample_size,
            "model_artifact_path": internal_summary.model_artifact_path,
            "summary_artifact_path": internal_summary.summary_artifact_path,
            "latency_mean_ms": internal_latency.total.mean_ms,
            "latency_p95_ms": internal_latency.total.p95_ms,
        },
        "pr_auc_delta": pr_auc_delta,
        "latency_delta_ms": latency_delta_ms,
        "recommendation": recommend_feature_group(
            pr_auc_delta=pr_auc_delta,
            latency_delta_ms=latency_delta_ms,
        ),
    }
    artifact_path = output_dir / "internal_feature_compare_summary.json"
    artifact_path.write_text(json.dumps(payload, indent=2))
    payload["artifact_path"] = str(artifact_path)
    return payload
