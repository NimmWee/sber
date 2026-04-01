from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from data.non_public_supervision import build_non_public_supervision_dataset
from eval.default_detector import build_default_detector_extractor, train_default_detector_head
from eval.error_analysis import analyze_prediction_errors
from eval.metrics import compute_pr_auc
from eval.public_benchmark import load_public_benchmark_examples
from eval.runner import RawLabeledExample
from models.head import TrainedLightGBMHead
from utils.script_helpers import write_json_artifact


def run_non_public_retraining_public_eval(
    *,
    public_dataset_path: str | Path,
    baseline_model_artifact_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
) -> dict:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    public_examples = load_public_benchmark_examples(public_dataset_path)
    supervision_dataset = build_non_public_supervision_dataset(
        public_eval_examples=public_examples
    )
    signal_cache_start = time.perf_counter()
    cached_train_examples = _cache_model_signals(
        examples=supervision_dataset.train_examples,
        token_stat_provider=token_stat_provider,
    )
    cached_dev_examples = _cache_model_signals(
        examples=supervision_dataset.dev_examples,
        token_stat_provider=token_stat_provider,
    )
    cached_public_examples = _cache_model_signals(
        examples=public_examples,
        token_stat_provider=token_stat_provider,
    )
    cached_signal_runtime_ms = (time.perf_counter() - signal_cache_start) * 1000.0

    extractor = build_default_detector_extractor()
    train_feature_rows = _extract_feature_rows(
        examples=cached_train_examples,
        extractor=extractor,
    )
    train_labels = [example.label for example in cached_train_examples]

    trained_head = train_default_detector_head(
        feature_rows=train_feature_rows,
        labels=train_labels,
    )
    trained_model_artifact_path = output_dir / "retrained_default_detector_head.json"
    trained_head.save(trained_model_artifact_path)

    baseline_head = TrainedLightGBMHead.load(baseline_model_artifact_path)
    before_summary = _evaluate_head_on_public_examples(
        examples=cached_public_examples,
        extractor=extractor,
        head=baseline_head,
    )
    after_summary = _evaluate_head_on_public_examples(
        examples=cached_public_examples,
        extractor=extractor,
        head=trained_head,
    )
    dev_summary = _evaluate_head_on_public_examples(
        examples=cached_dev_examples,
        extractor=extractor,
        head=trained_head,
    )

    bucket_deltas = {
        bucket_name: {
            "false_positive_delta": after_summary["bucket_summaries"][bucket_name].false_positive_count
            - before_summary["bucket_summaries"][bucket_name].false_positive_count,
            "false_negative_delta": after_summary["bucket_summaries"][bucket_name].false_negative_count
            - before_summary["bucket_summaries"][bucket_name].false_negative_count,
        }
        for bucket_name in (
            "numbers",
            "entity_like_tokens",
            "places",
            "long_responses",
        )
    }
    false_positive_increase = (
        after_summary["false_positive_count"] - before_summary["false_positive_count"]
    )
    false_negative_delta = (
        after_summary["false_negative_count"] - before_summary["false_negative_count"]
    )

    payload = {
        "dataset_summary": supervision_dataset.summary,
        "public_benchmark": {
            "before": before_summary,
            "after": after_summary,
            "bucket_deltas": bucket_deltas,
        },
        "recall_recovery": {
            "false_negatives_decreased": false_negative_delta < 0,
            "false_negative_delta": false_negative_delta,
            "false_positive_increase": false_positive_increase,
            "false_positive_increase_too_much": false_positive_increase > 25,
        },
        "dev_summary": dev_summary,
        "trained_model_artifact_path": str(trained_model_artifact_path),
        "cached_signal_runtime_ms": cached_signal_runtime_ms,
    }
    artifact_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="non_public_recovery_summary.json",
        payload=_json_safe_payload(payload),
    )
    payload["artifact_path"] = str(artifact_path)
    return payload


def _cache_model_signals(
    *,
    examples: list[RawLabeledExample],
    token_stat_provider,
) -> list[RawLabeledExample]:
    cached: list[RawLabeledExample] = []
    for example in examples:
        collected = token_stat_provider.collect_signals(
            prompt=example.prompt,
            response=example.response,
        )
        cached.append(
            RawLabeledExample(
                prompt=example.prompt,
                response=example.response,
                label=example.label,
                token_stats=collected.token_stats,
                internal_signal=collected.internal_signal,
            )
        )
    return cached


def _extract_feature_rows(
    *,
    examples: list[RawLabeledExample],
    extractor,
) -> list[dict[str, float]]:
    return [
        dict(
            extractor.extract(
                prompt=example.prompt,
                response=example.response,
                token_stats=example.token_stats,
                internal_signal=example.internal_signal,
            )
        )
        for example in examples
    ]


def _evaluate_head_on_public_examples(
    *,
    examples: list[RawLabeledExample],
    extractor,
    head,
) -> dict:
    feature_rows = _extract_feature_rows(examples=examples, extractor=extractor)
    probabilities = head.predict_proba_batch(feature_rows)
    labels = [example.label for example in examples]
    pr_auc = compute_pr_auc(labels, probabilities)
    error_summary = analyze_prediction_errors(
        validation_examples=examples,
        probabilities=probabilities,
        pr_auc=pr_auc,
    )
    latency_start = time.perf_counter()
    for feature_row in feature_rows:
        head.predict_proba(feature_row)
    head_only_latency_ms = (
        ((time.perf_counter() - latency_start) * 1000.0) / len(feature_rows)
        if feature_rows
        else 0.0
    )
    return {
        "pr_auc": pr_auc,
        "sample_size": len(examples),
        "false_positive_count": error_summary.false_positive_count,
        "false_negative_count": error_summary.false_negative_count,
        "non_trivial_buckets": error_summary.non_trivial_buckets,
        "bucket_summaries": error_summary.bucket_summaries,
        "hardest_examples": error_summary.hardest_examples,
        "head_only_latency_mean_ms": head_only_latency_ms,
    }


def _json_safe_payload(payload: dict) -> dict:
    public_benchmark = payload["public_benchmark"]
    dev_summary = payload["dev_summary"]
    return {
        "dataset_summary": payload["dataset_summary"],
        "public_benchmark": {
            "before": _serialize_eval_summary(public_benchmark["before"]),
            "after": _serialize_eval_summary(public_benchmark["after"]),
            "bucket_deltas": public_benchmark["bucket_deltas"],
        },
        "recall_recovery": payload["recall_recovery"],
        "dev_summary": _serialize_eval_summary(dev_summary),
        "trained_model_artifact_path": payload["trained_model_artifact_path"],
        "cached_signal_runtime_ms": payload["cached_signal_runtime_ms"],
    }


def _serialize_eval_summary(summary: dict) -> dict:
    return {
        "pr_auc": summary["pr_auc"],
        "sample_size": summary["sample_size"],
        "false_positive_count": summary["false_positive_count"],
        "false_negative_count": summary["false_negative_count"],
        "non_trivial_buckets": summary["non_trivial_buckets"],
        "bucket_summaries": {
            name: asdict(bucket_summary)
            for name, bucket_summary in summary["bucket_summaries"].items()
        },
        "hardest_examples": [
            asdict(example) for example in summary["hardest_examples"]
        ],
        "head_only_latency_mean_ms": summary["head_only_latency_mean_ms"],
    }
