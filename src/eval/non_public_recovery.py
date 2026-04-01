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


DEFAULT_RECOVERY_BLEND_WEIGHT = 0.35


class _BlendedLightGBMHead:
    def __init__(
        self,
        *,
        baseline_head: TrainedLightGBMHead,
        recovery_head: TrainedLightGBMHead,
        recovery_blend_weight: float,
    ) -> None:
        self.baseline_head = baseline_head
        self.recovery_head = recovery_head
        self.recovery_blend_weight = recovery_blend_weight

    def predict_proba(self, features) -> float:
        baseline_probability = self.baseline_head.predict_proba(features)
        recovery_probability = self.recovery_head.predict_proba(features)
        return ((1.0 - self.recovery_blend_weight) * baseline_probability) + (
            self.recovery_blend_weight * recovery_probability
        )

    def predict_proba_batch(self, feature_rows):
        return [self.predict_proba(feature_row) for feature_row in feature_rows]


def run_non_public_retraining_public_eval(
    *,
    public_dataset_path: str | Path,
    baseline_model_artifact_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
    recovery_blend_weight: float = DEFAULT_RECOVERY_BLEND_WEIGHT,
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
        sample_weights=supervision_dataset.train_sample_weights,
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
        head=_BlendedLightGBMHead(
            baseline_head=baseline_head,
            recovery_head=trained_head,
            recovery_blend_weight=recovery_blend_weight,
        ),
    )
    dev_summary = _evaluate_head_on_public_examples(
        examples=cached_dev_examples,
        extractor=extractor,
        head=_BlendedLightGBMHead(
            baseline_head=baseline_head,
            recovery_head=trained_head,
            recovery_blend_weight=recovery_blend_weight,
        ),
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
    guardrails = _build_guardrails(
        before_summary=before_summary,
        after_summary=after_summary,
    )
    decision = _build_change_decision(
        before_summary=before_summary,
        after_summary=after_summary,
        guardrails=guardrails,
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
            "false_positive_increase_too_much": decision["false_positive_increase_too_much"],
        },
        "precision_change": after_summary["precision"] - before_summary["precision"],
        "guardrails": guardrails,
        "decision": decision,
        "dev_summary": dev_summary,
        "trained_model_artifact_path": str(trained_model_artifact_path),
        "cached_signal_runtime_ms": cached_signal_runtime_ms,
        "training_config": {
            "recovery_blend_weight": recovery_blend_weight,
            "train_sample_weight_sum": sum(supervision_dataset.train_sample_weights),
        },
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
    metrics = _binary_metrics(labels=labels, probabilities=probabilities)
    bucket_predicted_positive_rates = _bucket_predicted_positive_rates(
        examples=examples,
        probabilities=probabilities,
    )
    score_distribution = _score_distribution(labels=labels, probabilities=probabilities)
    return {
        "pr_auc": pr_auc,
        "sample_size": len(examples),
        "false_positive_count": error_summary.false_positive_count,
        "false_negative_count": error_summary.false_negative_count,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "predicted_positive_rate": metrics["predicted_positive_rate"],
        "bucket_predicted_positive_rates": bucket_predicted_positive_rates,
        "score_distribution": score_distribution,
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
        "precision_change": payload["precision_change"],
        "guardrails": payload["guardrails"],
        "decision": payload["decision"],
        "dev_summary": _serialize_eval_summary(dev_summary),
        "trained_model_artifact_path": payload["trained_model_artifact_path"],
        "cached_signal_runtime_ms": payload["cached_signal_runtime_ms"],
        "training_config": payload["training_config"],
    }


def _serialize_eval_summary(summary: dict) -> dict:
    return {
        "pr_auc": summary["pr_auc"],
        "sample_size": summary["sample_size"],
        "false_positive_count": summary["false_positive_count"],
        "false_negative_count": summary["false_negative_count"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "predicted_positive_rate": summary["predicted_positive_rate"],
        "bucket_predicted_positive_rates": summary["bucket_predicted_positive_rates"],
        "score_distribution": summary["score_distribution"],
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


def _build_change_decision(
    *,
    before_summary: dict,
    after_summary: dict,
    guardrails: dict[str, bool],
) -> dict:
    false_positive_limit = before_summary["false_positive_count"] + max(
        10,
        int(before_summary["false_positive_count"] * 0.2),
    )
    pr_auc_improved = after_summary["pr_auc"] > before_summary["pr_auc"]
    false_negatives_decreased = (
        after_summary["false_negative_count"] < before_summary["false_negative_count"]
    )
    false_positives_controlled = (
        after_summary["false_positive_count"] <= false_positive_limit
    )

    rejection_reasons: list[str] = []
    if not pr_auc_improved:
        rejection_reasons.append("PR-AUC did not improve")
    if not false_negatives_decreased:
        rejection_reasons.append("false negatives did not decrease")
    if not false_positives_controlled:
        rejection_reasons.append("false positives increased beyond the control limit")
    if guardrails["predicted_positive_rate_drop_too_far"]:
        rejection_reasons.append("predicted positive rate dropped too far")
    if guardrails["recall_collapsed"]:
        rejection_reasons.append("recall collapsed")
    if guardrails["long_response_positive_rate_collapsed"]:
        rejection_reasons.append("long-response positive prediction rate collapsed")
    if guardrails["score_distribution_compressed"]:
        rejection_reasons.append("score distribution compressed too strongly")

    return {
        "accept_change": (
            pr_auc_improved
            and false_negatives_decreased
            and false_positives_controlled
            and not any(guardrails.values())
        ),
        "rejection_reason": (
            "; ".join(rejection_reasons) if rejection_reasons else None
        ),
        "false_positive_limit": false_positive_limit,
        "false_positive_increase_too_much": not false_positives_controlled,
    }


def _build_guardrails(
    *,
    before_summary: dict,
    after_summary: dict,
) -> dict:
    long_before = before_summary["bucket_predicted_positive_rates"]["long_responses"]
    long_after = after_summary["bucket_predicted_positive_rates"]["long_responses"]
    before_spread = (
        before_summary["score_distribution"]["overall"]["q90"]
        - before_summary["score_distribution"]["overall"]["q10"]
    )
    after_spread = (
        after_summary["score_distribution"]["overall"]["q90"]
        - after_summary["score_distribution"]["overall"]["q10"]
    )
    return {
        "predicted_positive_rate_drop_too_far": (
            after_summary["predicted_positive_rate"]
            < before_summary["predicted_positive_rate"] * 0.7
        ),
        "recall_collapsed": after_summary["recall"] < before_summary["recall"] * 0.7,
        "long_response_positive_rate_collapsed": long_after < long_before * 0.7,
        "score_distribution_compressed": after_spread < before_spread * 0.7,
    }


def _binary_metrics(
    *,
    labels: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    predicted_positive_count = 0

    for label, probability in zip(labels, probabilities):
        predicted_label = 1 if probability >= threshold else 0
        if predicted_label == 1:
            predicted_positive_count += 1
        if predicted_label == 1 and label == 1:
            true_positives += 1
        elif predicted_label == 1 and label == 0:
            false_positives += 1
        elif predicted_label == 0 and label == 1:
            false_negatives += 1

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return {
        "precision": (
            true_positives / precision_denominator
            if precision_denominator
            else 0.0
        ),
        "recall": (
            true_positives / recall_denominator
            if recall_denominator
            else 0.0
        ),
        "predicted_positive_rate": (
            predicted_positive_count / len(labels) if labels else 0.0
        ),
    }


def _score_distribution(
    *,
    labels: list[int],
    probabilities: list[float],
) -> dict[str, dict[str, float]]:
    overall = _distribution_stats(probabilities)
    hallucination_scores = [
        probability for label, probability in zip(labels, probabilities) if label == 1
    ]
    non_hallucination_scores = [
        probability for label, probability in zip(labels, probabilities) if label == 0
    ]
    return {
        "overall": overall,
        "hallucination": _distribution_stats(hallucination_scores),
        "non_hallucination": _distribution_stats(non_hallucination_scores),
    }


def _distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "mean": sum(ordered) / len(ordered),
        "q10": _quantile(ordered, 0.10),
        "q50": _quantile(ordered, 0.50),
        "q90": _quantile(ordered, 0.90),
    }


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * weight


def _bucket_predicted_positive_rates(
    *,
    examples: list[RawLabeledExample],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    total_counts = {
        "numbers": 0,
        "entity_like_tokens": 0,
        "places": 0,
        "long_responses": 0,
    }
    predicted_positive_counts = {bucket_name: 0 for bucket_name in total_counts}

    for example, probability in zip(examples, probabilities):
        buckets = _summary_buckets(example)
        predicted_positive = probability >= threshold
        for bucket_name in buckets:
            total_counts[bucket_name] += 1
            if predicted_positive:
                predicted_positive_counts[bucket_name] += 1

    return {
        bucket_name: (
            predicted_positive_counts[bucket_name] / total_counts[bucket_name]
            if total_counts[bucket_name]
            else 0.0
        )
        for bucket_name in total_counts
    }


def _summary_buckets(example: RawLabeledExample) -> list[str]:
    buckets: list[str] = []
    if any(character.isdigit() for character in example.response):
        buckets.append("numbers")
    if sum(token[:1].isupper() for token in example.response.split()) >= 2:
        buckets.append("entity_like_tokens")
    if any(
        cue in example.prompt.lower()
        for cue in ("capital", "city", "country", "located", "location", "place")
    ):
        buckets.append("places")
    if len(example.response.split()) > 8:
        buckets.append("long_responses")
    return buckets
