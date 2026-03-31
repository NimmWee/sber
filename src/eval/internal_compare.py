import json
from dataclasses import replace
from pathlib import Path

from eval.ablation import filter_feature_rows, recommend_feature_group
from eval.metrics import compute_pr_auc
from eval.default_detector import (
    build_default_detector_extractor,
    build_default_detector_feature_allowlist,
    filter_default_detector_rows,
)
from eval.runner import RawExampleEvaluationDataset, RawLabeledExample, TrainValidationEvaluationRunner
from features.extractor import StructuralFeatureExtractor
from inference.token_stats import TokenStatProvider
from models.head import TrainedLogisticRegressionHead, train_logistic_regression_head
from utils.latency import LatencyBenchmarkConfig, benchmark_single_example_latency_with_provider


def compare_base_vs_internal_features(
    *,
    train_examples: list[RawLabeledExample],
    validation_examples: list[RawLabeledExample],
    base_provider: TokenStatProvider | None,
    internal_provider: TokenStatProvider,
    artifact_dir: str | Path,
    latency_repeat_count: int = 20,
) -> dict:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_provider is None:
        if not hasattr(internal_provider, "share_backend") or not hasattr(
            internal_provider, "config"
        ):
            raise ValueError(
                "base_provider is required when internal_provider cannot share its backend"
            )
        base_provider = internal_provider.share_backend(
            config=replace(internal_provider.config, enable_internal_features=False)
        )

    baseline_runner = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=build_default_detector_extractor(),
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
                token_feature_groups=("base_token_uncertainty",),
            ),
            token_stat_provider=internal_provider,
        ),
        artifact_dir=output_dir / "internal",
    )
    internal_split = internal_runner.dataset.load_split()
    filtered_train_features = _filter_internal_probe_rows(
        feature_rows=internal_split.train_features,
    )
    filtered_validation_features = _filter_internal_probe_rows(
        feature_rows=internal_split.validation_features,
    )
    internal_head = train_logistic_regression_head(
        filtered_train_features,
        internal_split.train_labels,
    )
    internal_probabilities = internal_head.predict_proba_batch(filtered_validation_features)
    internal_pr_auc = compute_pr_auc(
        internal_split.validation_labels,
        internal_probabilities,
    )
    internal_model_artifact_path = output_dir / "internal" / "logistic_head.json"
    internal_summary_artifact_path = output_dir / "internal" / "eval_summary.json"
    internal_model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    internal_head.save(internal_model_artifact_path)
    internal_summary_artifact_path.write_text(
        json.dumps(
            {
                "pr_auc": internal_pr_auc,
                "sample_size": len(internal_split.validation_labels),
                "model_artifact_path": str(internal_model_artifact_path),
                "summary_artifact_path": str(internal_summary_artifact_path),
            },
            indent=2,
        )
    )
    internal_summary = {
        "pr_auc": internal_pr_auc,
        "sample_size": len(internal_split.validation_labels),
        "model_artifact_path": str(internal_model_artifact_path),
        "summary_artifact_path": str(internal_summary_artifact_path),
    }

    latency_example = validation_examples[0]
    baseline_latency = benchmark_single_example_latency_with_provider(
        prompt=latency_example.prompt,
        response=latency_example.response,
        extractor=build_default_detector_extractor(),
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
            token_feature_groups=("base_token_uncertainty",),
        ),
        head=internal_head,
        token_stat_provider=internal_provider,
        config=LatencyBenchmarkConfig(
            repeat_count=latency_repeat_count,
            artifact_dir=output_dir / "internal_latency",
        ),
    )

    pr_auc_delta = internal_summary["pr_auc"] - baseline_summary.pr_auc
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
            "pr_auc": internal_summary["pr_auc"],
            "sample_size": internal_summary["sample_size"],
            "model_artifact_path": internal_summary["model_artifact_path"],
            "summary_artifact_path": internal_summary["summary_artifact_path"],
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


def _filter_internal_probe_rows(
    *,
    feature_rows,
) -> list[dict[str, float]]:
    default_allowlist = build_default_detector_feature_allowlist(feature_rows=feature_rows)
    internal_feature_names = {
        feature_name
        for feature_row in feature_rows
        for feature_name in feature_row
        if feature_name.startswith("internal_")
    }
    return filter_feature_rows(
        feature_rows=feature_rows,
        allowed_features=default_allowlist | internal_feature_names,
    )
