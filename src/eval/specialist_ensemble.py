from __future__ import annotations

import re

from features.extractor import InternalModelSignal, TokenUncertaintyStat


NUMBER_PATTERN = re.compile(r"\d")
TITLECASE_PATTERN = re.compile(r"^[A-Z][a-zA-Z-]+$")


def extract_specialist_features(
    *,
    prompt: str,
    response: str,
    token_stats: list[TokenUncertaintyStat] | None,
    internal_signal: InternalModelSignal | None,
) -> dict[str, float]:
    stats = token_stats or []
    tokens = [stat.token for stat in stats]
    token_count = max(len(tokens), 1)

    numeric_indices = [index for index, token in enumerate(tokens) if NUMBER_PATTERN.search(token)]
    entity_indices = [index for index, token in enumerate(tokens) if TITLECASE_PATTERN.match(_normalize_token(token))]
    logprobs = [float(stat.logprob) for stat in stats]
    entropies = [float(stat.entropy) for stat in stats]
    margins = [float(stat.top1_top2_margin) for stat in stats]

    return {
        "specialist_numeric_density": len(numeric_indices) / token_count,
        "specialist_numeric_span_count": float(len(numeric_indices)),
        "specialist_numeric_margin_mean": _subset_mean(margins, numeric_indices),
        "specialist_numeric_margin_min": _subset_min(margins, numeric_indices),
        "specialist_numeric_entropy_mean": _subset_mean(entropies, numeric_indices),
        "specialist_numeric_logprob_asymmetry": _subset_mean(logprobs, numeric_indices) - _mean(logprobs),
        "specialist_numeric_tail_suspicion": _tail_suspicion(indices=numeric_indices, margins=margins, entropies=entropies),
        "specialist_entity_density": len(entity_indices) / token_count,
        "specialist_entity_margin_mean": _subset_mean(margins, entity_indices),
        "specialist_entity_margin_min": _subset_min(margins, entity_indices),
        "specialist_entity_entropy_mean": _subset_mean(entropies, entity_indices),
        "specialist_entity_confidence_dip": _mean(margins) - _subset_mean(margins, entity_indices),
        "specialist_entity_segment_suspicion": _segment_suspicion(entity_indices, entropies, margins),
        "specialist_long_length_bucket": _length_bucket(response),
        "specialist_long_entropy_drift": _tail_mean(entropies) - _head_mean(entropies),
        "specialist_long_margin_drift": _head_mean(margins) - _tail_mean(margins),
        "specialist_long_segment_variance": _segment_variance(entropies),
        "specialist_long_max_suspicious_segment_score": _max_suspicious_segment_score(entropies, margins),
        "specialist_long_longest_suspicious_run": float(_longest_suspicious_run(entropies, margins)),
        "specialist_internal_disagreement_hint": float(internal_signal.layer_disagreement_mean) if internal_signal else 0.0,
        "specialist_internal_consistency_hint": float(internal_signal.early_late_layer_consistency) if internal_signal else 0.0,
        "specialist_bucket_hint_numbers": 1.0 if ("how many" in prompt.lower() or any(character.isdigit() for character in response)) else 0.0,
        "specialist_bucket_hint_entities": 1.0 if entity_indices else 0.0,
        "specialist_bucket_hint_long": 1.0 if len(response.split()) > 18 else 0.0,
    }


def build_fusion_feature_row(
    *,
    baseline_score: float,
    numeric_score: float,
    entity_score: float,
    long_score: float,
    specialist_features: dict[str, float],
) -> dict[str, float]:
    row = {
        "baseline_score": float(baseline_score),
        "numeric_specialist_score": float(numeric_score),
        "entity_specialist_score": float(entity_score),
        "long_specialist_score": float(long_score),
    }
    row.update({name: float(value) for name, value in specialist_features.items()})
    return row


def select_specialist_feature_subset(
    *,
    specialist_rows: list[dict[str, float]],
    prefix: str,
) -> list[dict[str, float]]:
    return [
        {
            name: value
            for name, value in row.items()
            if name.startswith(prefix) or name in {"specialist_internal_disagreement_hint", "specialist_internal_consistency_hint"}
        }
        for row in specialist_rows
    ]


def build_single_specialist_blend(
    *,
    baseline_scores: list[float],
    specialist_scores: list[float],
    blend_weight: float = 0.25,
) -> list[float]:
    return [
        ((1.0 - blend_weight) * baseline_score) + (blend_weight * specialist_score)
        for baseline_score, specialist_score in zip(baseline_scores, specialist_scores)
    ]


def build_all_specialist_blend(
    *,
    baseline_scores: list[float],
    numeric_scores: list[float],
    entity_scores: list[float],
    long_scores: list[float],
) -> list[float]:
    blended: list[float] = []
    for baseline_score, numeric_score, entity_score, long_score in zip(
        baseline_scores,
        numeric_scores,
        entity_scores,
        long_scores,
    ):
        blended.append(
            (0.55 * baseline_score)
            + (0.15 * numeric_score)
            + (0.15 * entity_score)
            + (0.15 * long_score)
        )
    return blended


def _normalize_token(token: str) -> str:
    return token.strip("Ġ▁ ,.;:!?()[]{}\"'")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _subset_mean(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return _mean(selected)


def _subset_min(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return min(selected) if selected else 0.0


def _tail_suspicion(*, indices: list[int], margins: list[float], entropies: list[float]) -> float:
    if not indices:
        return 0.0
    split = max(1, len(margins) // 2)
    tail_indices = [index for index in indices if index >= split]
    if not tail_indices:
        return 0.0
    tail_entropy = _subset_mean(entropies, tail_indices)
    tail_margin = _subset_mean(margins, tail_indices)
    return tail_entropy - tail_margin


def _segment_suspicion(indices: list[int], entropies: list[float], margins: list[float]) -> float:
    if not indices:
        return 0.0
    segment_size = max(1, len(entropies) // 3)
    segment_scores: list[float] = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_indices = [index for index in indices if start <= index < end]
        if not segment_indices:
            continue
        segment_scores.append(
            _subset_mean(entropies, segment_indices) - _subset_mean(margins, segment_indices)
        )
    return max(segment_scores) if segment_scores else 0.0


def _length_bucket(response: str) -> float:
    token_count = len(response.split())
    if token_count <= 8:
        return 0.0
    if token_count <= 20:
        return 1.0
    return 2.0


def _tail_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    tail = values[-max(1, len(values) // 3) :]
    return _mean(tail)


def _head_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    head = values[: max(1, len(values) // 3)]
    return _mean(head)


def _segment_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    segment_size = max(1, len(values) // 3)
    means = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(values) if segment_index == 2 else min(len(values), start + segment_size)
        segment = values[start:end]
        if segment:
            means.append(_mean(segment))
    overall = _mean(means)
    return _mean([(value - overall) ** 2 for value in means])


def _max_suspicious_segment_score(entropies: list[float], margins: list[float]) -> float:
    if not entropies or not margins:
        return 0.0
    segment_size = max(1, len(entropies) // 3)
    scores = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_entropies = entropies[start:end]
        segment_margins = margins[start:end]
        if segment_entropies and segment_margins:
            scores.append(_mean(segment_entropies) - _mean(segment_margins))
    return max(scores) if scores else 0.0


def _longest_suspicious_run(entropies: list[float], margins: list[float]) -> int:
    longest = 0
    current = 0
    for entropy, margin in zip(entropies, margins):
        if entropy >= 0.7 or margin <= 0.15:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
