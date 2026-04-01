from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

from eval.ablation import build_feature_allowlist, filter_feature_rows, recommend_feature_group
from eval.default_detector import DEFAULT_TOKEN_FEATURE_GROUPS
from eval.error_analysis import analyze_prediction_errors
from eval.metrics import compute_pr_auc
from eval.public_benchmark import load_public_benchmark_examples
from eval.runner import RawLabeledExample
from features.extractor import StructuralFeatureExtractor
from models.head import train_lightgbm_head, train_logistic_regression_head
from utils.script_helpers import write_json_artifact


DEFAULT_LATENCY_REPEAT_COUNT = 5


def run_public_benchmark_ablation(
    *,
    dataset_path: str | Path,
    train_examples: list[RawLabeledExample],
    token_stat_provider,
    artifact_dir: str | Path,
    latency_repeat_count: int = DEFAULT_LATENCY_REPEAT_COUNT,
) -> dict:
    validation_examples = load_public_benchmark_examples(dataset_path)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_provider = _resolve_internal_provider(token_stat_provider)
    cache_start = time.perf_counter()
    cached_train_examples = _cache_model_signals(
        examples=train_examples,
        token_stat_provider=signal_provider,
    )
    cached_validation_examples = _cache_model_signals(
        examples=validation_examples,
        token_stat_provider=signal_provider,
    )
    signal_collection_runtime_ms = (time.perf_counter() - cache_start) * 1000.0
    cached_signal_artifact_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="cached_model_signals.json",
        payload={
            "train": [_serialize_cached_example(example) for example in cached_train_examples],
            "validation": [
                _serialize_cached_example(example) for example in cached_validation_examples
            ],
        },
    )

    base_variant = _evaluate_variant(
        name="base_token_uncertainty",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            token_feature_groups=DEFAULT_TOKEN_FEATURE_GROUPS,
        ),
        allowed_token_groups=("base_token_uncertainty",),
        artifact_dir=output_dir / "base_token_uncertainty",
        latency_repeat_count=latency_repeat_count,
    )
    extended_variant = _evaluate_variant(
        name="extended_token_uncertainty",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            token_feature_groups=(
                "base_token_uncertainty",
                "variance_std",
                "segment_summaries",
                "span_tail_rates",
                "specialized_tokens",
            ),
        ),
        allowed_token_groups=(
            "base_token_uncertainty",
            "variance_std",
            "segment_summaries",
            "span_tail_rates",
            "specialized_tokens",
        ),
        artifact_dir=output_dir / "extended_token_uncertainty",
        latency_repeat_count=latency_repeat_count,
    )
    internal_variant = _evaluate_variant(
        name="internal_features",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            enable_internal_features=True,
            token_feature_groups=("base_token_uncertainty",),
        ),
        allowed_token_groups=("base_token_uncertainty",),
        include_internal_features=True,
        artifact_dir=output_dir / "internal_features",
        latency_repeat_count=latency_repeat_count,
    )

    variants = {
        base_variant["name"]: base_variant,
        extended_variant["name"]: extended_variant,
        internal_variant["name"]: internal_variant,
    }

    stronger_head_variant = _evaluate_variant(
        name="extended_token_uncertainty_stronger_head",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            token_feature_groups=(
                "base_token_uncertainty",
                "variance_std",
                "segment_summaries",
                "span_tail_rates",
                "specialized_tokens",
            ),
        ),
        allowed_token_groups=(
            "base_token_uncertainty",
            "variance_std",
            "segment_summaries",
            "span_tail_rates",
            "specialized_tokens",
        ),
        artifact_dir=output_dir / "extended_token_uncertainty_stronger_head",
        latency_repeat_count=latency_repeat_count,
        head_epochs=600,
        head_learning_rate=0.08,
    )
    variants[stronger_head_variant["name"]] = stronger_head_variant
    lightgbm_variant = _evaluate_variant(
        name="internal_features_lightgbm",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            enable_internal_features=True,
            token_feature_groups=("base_token_uncertainty",),
        ),
        allowed_token_groups=("base_token_uncertainty",),
        include_internal_features=True,
        artifact_dir=output_dir / "internal_features_lightgbm",
        latency_repeat_count=latency_repeat_count,
        head_kind="lightgbm",
    )
    variants[lightgbm_variant["name"]] = lightgbm_variant
    improved_lightgbm_variant = _evaluate_variant(
        name="improved_internal_features_lightgbm",
        train_examples=cached_train_examples,
        validation_examples=cached_validation_examples,
        token_stat_provider=signal_provider,
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            enable_internal_features=True,
            enable_compact_internal_enhancements=True,
            token_feature_groups=("base_token_uncertainty",),
        ),
        allowed_token_groups=("base_token_uncertainty",),
        include_internal_features=True,
        artifact_dir=output_dir / "improved_internal_features_lightgbm",
        latency_repeat_count=latency_repeat_count,
        head_kind="lightgbm",
    )
    variants[improved_lightgbm_variant["name"]] = improved_lightgbm_variant

    best_variant_name = max(
        variants,
        key=lambda name: (variants[name]["pr_auc"], -variants[name]["latency_total_mean_ms"]),
    )
    for name, variant in variants.items():
        variant["pr_auc_delta_vs_base"] = variant["pr_auc"] - base_variant["pr_auc"]
        variant["latency_delta_vs_base_ms"] = (
            variant["latency_total_mean_ms"] - base_variant["latency_total_mean_ms"]
        )
        variant["recommendation"] = recommend_feature_group(
            pr_auc_delta=variant["pr_auc_delta_vs_base"],
            latency_delta_ms=variant["latency_delta_vs_base_ms"],
        )

    payload = {
        "dataset_path": str(Path(dataset_path)),
        "sample_size": len(cached_validation_examples),
        "variants": variants,
        "best_variant": best_variant_name,
        "cached_signal_artifact_path": str(cached_signal_artifact_path),
        "signal_collection_runtime_ms": signal_collection_runtime_ms,
        "estimated_signal_runtime_improvement_ms": signal_collection_runtime_ms
        * (len(variants) - 1),
    }
    artifact_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="public_benchmark_ablation_summary.json",
        payload=payload,
    )
    payload["artifact_path"] = str(artifact_path)
    return payload


def _evaluate_variant(
    *,
    name: str,
    train_examples: list[RawLabeledExample],
    validation_examples: list[RawLabeledExample],
    token_stat_provider,
    extractor: StructuralFeatureExtractor,
    allowed_token_groups: tuple[str, ...],
    artifact_dir: Path,
    latency_repeat_count: int,
    include_internal_features: bool = False,
    head_epochs: int = 250,
    head_learning_rate: float = 0.1,
    head_kind: str = "logistic",
) -> dict:
    train_rows, train_labels = _prepare_feature_rows(
        examples=train_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
        include_internal_features=include_internal_features,
    )
    validation_rows, validation_labels = _prepare_feature_rows(
        examples=validation_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
        include_internal_features=include_internal_features,
    )

    structural_feature_names = _structural_feature_names(train_rows + validation_rows)
    allowed_features = build_feature_allowlist(
        structural_feature_names=structural_feature_names,
        enabled_groups=allowed_token_groups,
    )
    if include_internal_features:
        allowed_features |= {
            feature_name
            for row in train_rows + validation_rows
            for feature_name in row
            if feature_name.startswith("internal_")
        }

    filtered_train_rows = filter_feature_rows(
        feature_rows=train_rows,
        allowed_features=allowed_features,
    )
    filtered_validation_rows = filter_feature_rows(
        feature_rows=validation_rows,
        allowed_features=allowed_features,
    )

    if head_kind == "lightgbm":
        head = train_lightgbm_head(filtered_train_rows, train_labels)
    else:
        head = train_logistic_regression_head(
            filtered_train_rows,
            train_labels,
            epochs=head_epochs,
            learning_rate=head_learning_rate,
        )
    probabilities = head.predict_proba_batch(filtered_validation_rows)
    pr_auc = compute_pr_auc(validation_labels, probabilities)
    error_summary = analyze_prediction_errors(
        validation_examples=validation_examples,
        probabilities=probabilities,
        pr_auc=pr_auc,
    )

    model_artifact_path = artifact_dir / "logistic_head.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    head.save(model_artifact_path)
    latency_total_mean_ms = _measure_variant_latency(
        example=validation_examples[0],
        extractor=extractor,
        head=head,
        repeat_count=latency_repeat_count,
    )
    summary_artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="eval_summary.json",
        payload={
            "name": name,
            "pr_auc": pr_auc,
            "sample_size": len(validation_examples),
            "latency_total_mean_ms": latency_total_mean_ms,
            "head_kind": head_kind,
            "false_positive_count": error_summary.false_positive_count,
            "false_negative_count": error_summary.false_negative_count,
            "non_trivial_buckets": error_summary.non_trivial_buckets,
        },
    )
    return {
        "name": name,
        "pr_auc": pr_auc,
        "sample_size": len(validation_examples),
        "latency_total_mean_ms": latency_total_mean_ms,
        "head_kind": head_kind,
        "false_positive_count": error_summary.false_positive_count,
        "false_negative_count": error_summary.false_negative_count,
        "non_trivial_buckets": error_summary.non_trivial_buckets,
        "hardest_examples": [asdict(example) for example in error_summary.hardest_examples],
        "summary_artifact_path": str(summary_artifact_path),
        "model_artifact_path": str(model_artifact_path),
    }


def _prepare_feature_rows(
    *,
    examples: list[RawLabeledExample],
    extractor: StructuralFeatureExtractor,
    token_stat_provider,
    include_internal_features: bool,
) -> tuple[list[dict[str, float]], list[int]]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for example in examples:
        token_stats = example.token_stats
        internal_signal = example.internal_signal
        if include_internal_features and (
            token_stats is None or internal_signal is None
        ):
            token_stats, internal_signal = _collect_signals(
                token_stat_provider=token_stat_provider,
                prompt=example.prompt,
                response=example.response,
            )
        elif token_stats is None:
            token_stats = token_stat_provider.collect(
                prompt=example.prompt,
                response=example.response,
            )
        rows.append(
            dict(
                extractor.extract(
                    prompt=example.prompt,
                    response=example.response,
                    token_stats=token_stats,
                    internal_signal=internal_signal,
                )
            )
        )
        labels.append(example.label)
    return rows, labels


def _collect_signals(*, token_stat_provider, prompt: str, response: str):
    if hasattr(token_stat_provider, "collect_signals"):
        collected = token_stat_provider.collect_signals(prompt=prompt, response=response)
        return collected.token_stats, collected.internal_signal
    return token_stat_provider.collect(prompt=prompt, response=response), None


def _resolve_internal_provider(token_stat_provider):
    if hasattr(token_stat_provider, "share_backend") and hasattr(token_stat_provider, "config"):
        from dataclasses import replace

        return token_stat_provider.share_backend(
            config=replace(token_stat_provider.config, enable_internal_features=True)
        )
    return token_stat_provider


def _cache_model_signals(
    *,
    examples: list[RawLabeledExample],
    token_stat_provider,
) -> list[RawLabeledExample]:
    cached_examples: list[RawLabeledExample] = []
    for example in examples:
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=example.prompt,
            response=example.response,
        )
        cached_examples.append(
            RawLabeledExample(
                prompt=example.prompt,
                response=example.response,
                label=example.label,
                token_stats=token_stats,
                internal_signal=internal_signal,
            )
        )
    return cached_examples


def _serialize_cached_example(example: RawLabeledExample) -> dict:
    return {
        "prompt": example.prompt,
        "response": example.response,
        "label": example.label,
        "token_stats": [
            {
                "token": stat.token,
                "logprob": stat.logprob,
                "entropy": stat.entropy,
                "top1_top2_margin": stat.top1_top2_margin,
            }
            for stat in (example.token_stats or [])
        ],
        "internal_signal": (
            asdict(example.internal_signal) if example.internal_signal is not None else None
        ),
    }


def _measure_variant_latency(
    *,
    example: RawLabeledExample,
    extractor: StructuralFeatureExtractor,
    head,
    repeat_count: int,
) -> float:
    samples_ms: list[float] = []
    for _ in range(repeat_count):
        start = time.perf_counter()
        features = extractor.extract(
            prompt=example.prompt,
            response=example.response,
            token_stats=example.token_stats,
            internal_signal=example.internal_signal,
        )
        head.predict_proba(features)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return sum(samples_ms) / len(samples_ms) if samples_ms else 0.0


def _structural_feature_names(feature_rows: list[dict[str, float]]) -> set[str]:
    return {
        feature_name
        for row in feature_rows
        for feature_name in row
        if not feature_name.startswith("token_") and not feature_name.startswith("internal_")
    }
