from __future__ import annotations

import re

from features.extractor import InternalModelSignal, TokenUncertaintyStat


NUMBER_PATTERN = re.compile(r"\d")
TITLECASE_PATTERN = re.compile(r"^[A-Z][a-zA-Z-]+$")

NUMERIC_CORE_FEATURES = (
    "specialist_numeric_density",
    "specialist_numeric_span_count",
    "specialist_numeric_margin_mean",
    "specialist_numeric_margin_min",
    "specialist_numeric_entropy_mean",
    "specialist_numeric_logprob_asymmetry",
    "specialist_numeric_tail_suspicion",
)
ENTITY_CORE_FEATURES = (
    "specialist_entity_density",
    "specialist_entity_margin_mean",
    "specialist_entity_margin_min",
    "specialist_entity_entropy_mean",
    "specialist_entity_confidence_dip",
    "specialist_entity_segment_suspicion",
)
LONG_CORE_FEATURES = (
    "specialist_long_length_bucket",
    "specialist_long_entropy_drift",
    "specialist_long_margin_drift",
    "specialist_long_segment_variance",
    "specialist_long_max_suspicious_segment_score",
    "specialist_long_longest_suspicious_run",
)
CONSISTENCY_CORE_FEATURES = (
    "specialist_consistency_segment_disagreement",
    "specialist_consistency_segment_range",
    "specialist_consistency_prompt_alignment",
    "specialist_consistency_prompt_drift",
    "specialist_consistency_local_conflict_rate",
)
STABILITY_CORE_FEATURES = (
    "specialist_stability_logprob_shift",
    "specialist_stability_entropy_shift",
    "specialist_stability_margin_shift",
    "specialist_stability_instability_rate",
)
SHARED_SPECIALIST_HINT_FEATURES = (
    "specialist_internal_disagreement_hint",
    "specialist_internal_consistency_hint",
)


def extract_specialist_features(
    *,
    prompt: str,
    response: str,
    token_stats: list[TokenUncertaintyStat] | None,
    internal_signal: InternalModelSignal | None,
    perturbed_token_stats: list[TokenUncertaintyStat] | None = None,
    perturbed_internal_signal: InternalModelSignal | None = None,
) -> dict[str, float]:
    stats = token_stats or []
    tokens = [stat.token for stat in stats]
    token_count = max(len(tokens), 1)

    numeric_indices = [index for index, token in enumerate(tokens) if NUMBER_PATTERN.search(token)]
    entity_indices = [index for index, token in enumerate(tokens) if TITLECASE_PATTERN.match(_normalize_token(token))]
    logprobs = [float(stat.logprob) for stat in stats]
    entropies = [float(stat.entropy) for stat in stats]
    margins = [float(stat.top1_top2_margin) for stat in stats]
    sliding_entropy_scores = _sliding_window_scores(entropies)
    sliding_margin_scores = _sliding_window_scores(margins)
    global_entropy_mean = _mean(entropies)
    global_margin_mean = _mean(margins)
    segment_suspicion_scores = _segment_suspicion_scores(entropies, margins)
    segment_prompt_alignment_scores = _segment_prompt_alignment_scores(prompt, tokens)
    perturbed_stats = perturbed_token_stats or []
    perturbed_logprobs = [float(stat.logprob) for stat in perturbed_stats]
    perturbed_entropies = [float(stat.entropy) for stat in perturbed_stats]
    perturbed_margins = [float(stat.top1_top2_margin) for stat in perturbed_stats]

    return {
        "specialist_numeric_density": len(numeric_indices) / token_count,
        "specialist_numeric_span_count": float(len(numeric_indices)),
        "specialist_numeric_margin_mean": _subset_mean(margins, numeric_indices),
        "specialist_numeric_margin_min": _subset_min(margins, numeric_indices),
        "specialist_numeric_entropy_mean": _subset_mean(entropies, numeric_indices),
        "specialist_numeric_logprob_asymmetry": _subset_mean(logprobs, numeric_indices) - _mean(logprobs),
        "specialist_numeric_tail_suspicion": _tail_suspicion(indices=numeric_indices, margins=margins, entropies=entropies),
        "specialist_numeric_small_delta_proxy": _small_numeric_delta_proxy(tokens, numeric_indices),
        "specialist_numeric_inconsistency_count": float(_numeric_inconsistency_count(tokens, numeric_indices)),
        "specialist_numeric_isolated_low_confidence_rate": _isolated_low_confidence_rate(
            indices=numeric_indices,
            logprobs=logprobs,
            entropies=entropies,
            margins=margins,
        ),
        "specialist_numeric_local_margin_dip": _local_margin_dip(
            indices=numeric_indices,
            margins=margins,
            global_margin_mean=global_margin_mean,
        ),
        "specialist_numeric_local_entropy_spike": _local_entropy_spike(
            indices=numeric_indices,
            entropies=entropies,
            global_entropy_mean=global_entropy_mean,
        ),
        "specialist_entity_density": len(entity_indices) / token_count,
        "specialist_entity_margin_mean": _subset_mean(margins, entity_indices),
        "specialist_entity_margin_min": _subset_min(margins, entity_indices),
        "specialist_entity_entropy_mean": _subset_mean(entropies, entity_indices),
        "specialist_entity_confidence_dip": _mean(margins) - _subset_mean(margins, entity_indices),
        "specialist_entity_segment_suspicion": _segment_suspicion(entity_indices, entropies, margins),
        "specialist_entity_margin_variance": _subset_variance(margins, entity_indices),
        "specialist_entity_local_margin_drop": _local_margin_dip(
            indices=entity_indices,
            margins=margins,
            global_margin_mean=global_margin_mean,
        ),
        "specialist_entity_local_instability": _local_instability(
            indices=entity_indices,
            entropies=entropies,
            margins=margins,
        ),
        "specialist_entity_segment_inconsistency": _segment_inconsistency(
            indices=entity_indices,
            entropies=entropies,
            margins=margins,
        ),
        "specialist_long_length_bucket": _length_bucket(response),
        "specialist_long_entropy_drift": _tail_mean(entropies) - _head_mean(entropies),
        "specialist_long_margin_drift": _head_mean(margins) - _tail_mean(margins),
        "specialist_long_segment_variance": _segment_variance(entropies),
        "specialist_long_max_suspicious_segment_score": _max_suspicious_segment_score(entropies, margins),
        "specialist_long_longest_suspicious_run": float(_longest_suspicious_run(entropies, margins)),
        "specialist_long_sliding_entropy_max": max(sliding_entropy_scores) if sliding_entropy_scores else 0.0,
        "specialist_long_sliding_entropy_variance": _variance(sliding_entropy_scores),
        "specialist_long_sliding_margin_min": min(sliding_margin_scores) if sliding_margin_scores else 0.0,
        "specialist_long_local_anomaly_peak": _local_anomaly_peak(entropies, margins),
        "specialist_long_inconsistent_span": float(_longest_local_anomaly_run(entropies, margins)),
        "specialist_consistency_segment_disagreement": _variance(segment_suspicion_scores),
        "specialist_consistency_segment_range": (
            max(segment_suspicion_scores) - min(segment_suspicion_scores)
            if segment_suspicion_scores
            else 0.0
        ),
        "specialist_consistency_prompt_alignment": _mean(segment_prompt_alignment_scores),
        "specialist_consistency_prompt_drift": _prompt_alignment_drift(segment_prompt_alignment_scores),
        "specialist_consistency_local_conflict_rate": _local_conflict_rate(entropies, margins),
        "specialist_stability_logprob_shift": abs(_mean(logprobs) - _mean(perturbed_logprobs)),
        "specialist_stability_entropy_shift": abs(_mean(entropies) - _mean(perturbed_entropies)),
        "specialist_stability_margin_shift": abs(_mean(margins) - _mean(perturbed_margins)),
        "specialist_stability_instability_rate": _pairwise_instability_rate(
            logprobs=logprobs,
            perturbed_logprobs=perturbed_logprobs,
            entropies=entropies,
            perturbed_entropies=perturbed_entropies,
            margins=margins,
            perturbed_margins=perturbed_margins,
            internal_signal=internal_signal,
            perturbed_internal_signal=perturbed_internal_signal,
        ),
        "specialist_internal_disagreement_hint": float(internal_signal.layer_disagreement_mean) if internal_signal else 0.0,
        "specialist_internal_consistency_hint": float(internal_signal.early_late_layer_consistency) if internal_signal else 0.0,
        "specialist_local_uncertainty_spike": _local_uncertainty_spike(entropies, global_entropy_mean),
        "specialist_local_margin_dip_contrast": _local_margin_dip_contrast(margins, global_margin_mean),
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
    selected_feature_names: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, float]]:
    selected = set(selected_feature_names or [])
    return [
        {
            name: value
            for name, value in row.items()
            if (
                (name.startswith(prefix) or name in SHARED_SPECIALIST_HINT_FEATURES)
                and (not selected or name in selected)
            )
        }
        for row in specialist_rows
    ]


def select_important_specialist_features(
    *,
    all_feature_names: list[str] | tuple[str, ...],
    feature_importance: list[dict[str, float]],
    required_feature_names: list[str] | tuple[str, ...],
    max_selected_features: int = 12,
    min_importance: float = 0.5,
) -> list[str]:
    available = set(all_feature_names)
    required = [name for name in required_feature_names if name in available]
    ranked_optional = [
        row["feature_name"]
        for row in sorted(
            feature_importance,
            key=lambda row: float(row["importance"]),
            reverse=True,
        )
        if row["feature_name"] in available
        and row["feature_name"] not in required
        and float(row["importance"]) >= min_importance
    ]
    selected = set(required)
    for name in ranked_optional:
        if len(selected) >= max_selected_features:
            break
        selected.add(name)
    if len(selected) == len(required):
        for row in sorted(
            feature_importance,
            key=lambda row: float(row["importance"]),
            reverse=True,
        ):
            name = row["feature_name"]
            if name in available and name not in selected:
                selected.add(name)
                if len(selected) >= min(max_selected_features, len(available)):
                    break
    return [name for name in all_feature_names if name in selected]


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
    consistency_scores: list[float] | None = None,
    stability_scores: list[float] | None = None,
) -> list[float]:
    blended: list[float] = []
    consistency_iterable = consistency_scores or [0.0] * len(baseline_scores)
    stability_iterable = stability_scores or [0.0] * len(baseline_scores)
    for baseline_score, numeric_score, entity_score, long_score, consistency_score, stability_score in zip(
        baseline_scores,
        numeric_scores,
        entity_scores,
        long_scores,
        consistency_iterable,
        stability_iterable,
    ):
        blended.append(
            (0.48 * baseline_score)
            + (0.11 * numeric_score)
            + (0.11 * entity_score)
            + (0.11 * long_score)
            + (0.10 * consistency_score)
            + (0.09 * stability_score)
        )
    return blended


def build_stability_specialist_blend(
    *,
    baseline_scores: list[float],
    stability_scores: list[float],
    blend_weight: float = 0.2,
) -> list[float]:
    return [
        ((1.0 - blend_weight) * baseline_score) + (blend_weight * stability_score)
        for baseline_score, stability_score in zip(baseline_scores, stability_scores)
    ]


def _normalize_token(token: str) -> str:
    return token.strip("Ġ▁ ,.;:!?()[]{}\"'")


def _normalize_text_token(token: str) -> str:
    return _normalize_token(token).lower()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _subset_mean(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return _mean(selected)


def _subset_min(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return min(selected) if selected else 0.0


def _subset_variance(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return _variance(selected)


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


def _segment_inconsistency(indices: list[int], entropies: list[float], margins: list[float]) -> float:
    if not indices:
        return 0.0
    segment_scores = _entity_region_scores(indices=indices, entropies=entropies, margins=margins)
    return _variance(segment_scores)


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


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return _mean([(value - mean) ** 2 for value in values])


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


def _segment_suspicion_scores(entropies: list[float], margins: list[float]) -> list[float]:
    if not entropies or not margins:
        return []
    segment_size = max(1, len(entropies) // 3)
    scores = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_entropies = entropies[start:end]
        segment_margins = margins[start:end]
        if segment_entropies and segment_margins:
            scores.append(_mean(segment_entropies) - _mean(segment_margins))
    return scores


def _segment_prompt_alignment_scores(prompt: str, response_tokens: list[str]) -> list[float]:
    if not response_tokens:
        return []
    prompt_terms = {
        normalized
        for normalized in (_normalize_text_token(token) for token in prompt.split())
        if normalized and len(normalized) >= 3
    }
    if not prompt_terms:
        return []
    segment_size = max(1, len(response_tokens) // 3)
    scores = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(response_tokens) if segment_index == 2 else min(len(response_tokens), start + segment_size)
        segment_tokens = [
            normalized
            for normalized in (_normalize_text_token(token) for token in response_tokens[start:end])
            if normalized
        ]
        if not segment_tokens:
            continue
        overlap_count = sum(1 for token in segment_tokens if token in prompt_terms)
        scores.append(overlap_count / len(segment_tokens))
    return scores


def _prompt_alignment_drift(alignment_scores: list[float]) -> float:
    if len(alignment_scores) < 2:
        return 0.0
    return max(alignment_scores) - alignment_scores[-1]


def _local_conflict_rate(entropies: list[float], margins: list[float]) -> float:
    if len(entropies) < 2 or len(margins) < 2:
        return 0.0
    conflicts = 0
    total = 0
    for left_index in range(len(entropies) - 1):
        entropy_jump = abs(entropies[left_index + 1] - entropies[left_index])
        margin_jump = abs(margins[left_index + 1] - margins[left_index])
        total += 1
        if entropy_jump + margin_jump >= 0.6:
            conflicts += 1
    return conflicts / total if total else 0.0


def _pairwise_instability_rate(
    *,
    logprobs: list[float],
    perturbed_logprobs: list[float],
    entropies: list[float],
    perturbed_entropies: list[float],
    margins: list[float],
    perturbed_margins: list[float],
    internal_signal: InternalModelSignal | None,
    perturbed_internal_signal: InternalModelSignal | None,
) -> float:
    shared_length = min(
        len(logprobs),
        len(perturbed_logprobs),
        len(entropies),
        len(perturbed_entropies),
        len(margins),
        len(perturbed_margins),
    )
    if shared_length <= 0:
        return 0.0
    unstable_positions = 0
    for index in range(shared_length):
        shift = (
            abs(logprobs[index] - perturbed_logprobs[index])
            + abs(entropies[index] - perturbed_entropies[index])
            + abs(margins[index] - perturbed_margins[index])
        )
        if shift >= 0.6:
            unstable_positions += 1
    instability_rate = unstable_positions / shared_length
    internal_shift = 0.0
    if internal_signal is not None and perturbed_internal_signal is not None:
        internal_shift = abs(
            float(internal_signal.layer_disagreement_mean)
            - float(perturbed_internal_signal.layer_disagreement_mean)
        )
    return instability_rate + internal_shift


def _sliding_window_scores(values: list[float], window_size: int = 3) -> list[float]:
    if not values:
        return []
    if len(values) <= window_size:
        return [_mean(values)]
    return [_mean(values[index : index + window_size]) for index in range(len(values) - window_size + 1)]


def _small_numeric_delta_proxy(tokens: list[str], numeric_indices: list[int]) -> float:
    numeric_values = []
    for index in numeric_indices:
        token = _normalize_token(tokens[index]).replace(",", "")
        try:
            numeric_values.append(float(token))
        except ValueError:
            continue
    if len(numeric_values) < 2:
        return 0.0
    deltas = [abs(right - left) for left, right in zip(numeric_values, numeric_values[1:])]
    if not deltas:
        return 0.0
    smallest_delta = min(deltas)
    return 1.0 / (1.0 + smallest_delta)


def _numeric_inconsistency_count(tokens: list[str], numeric_indices: list[int]) -> int:
    numeric_values = []
    for index in numeric_indices:
        token = _normalize_token(tokens[index]).replace(",", "")
        try:
            numeric_values.append(float(token))
        except ValueError:
            continue
    if len(numeric_values) < 2:
        return 0
    return sum(abs(right - left) > 1.0 for left, right in zip(numeric_values, numeric_values[1:]))


def _isolated_low_confidence_rate(
    *,
    indices: list[int],
    logprobs: list[float],
    entropies: list[float],
    margins: list[float],
) -> float:
    if not indices:
        return 0.0
    suspicious = 0
    for index in indices:
        if (
            index < len(logprobs)
            and index < len(entropies)
            and index < len(margins)
            and (logprobs[index] <= -1.0 or entropies[index] >= 0.8 or margins[index] <= 0.15)
        ):
            suspicious += 1
    return suspicious / len(indices)


def _local_margin_dip(
    *,
    indices: list[int],
    margins: list[float],
    global_margin_mean: float,
) -> float:
    if not indices or not margins:
        return 0.0
    local_means = []
    for index in indices:
        start = max(0, index - 1)
        end = min(len(margins), index + 2)
        local_means.append(_mean(margins[start:end]))
    return max(0.0, global_margin_mean - _mean(local_means))


def _local_entropy_spike(
    *,
    indices: list[int],
    entropies: list[float],
    global_entropy_mean: float,
) -> float:
    if not indices or not entropies:
        return 0.0
    local_means = []
    for index in indices:
        start = max(0, index - 1)
        end = min(len(entropies), index + 2)
        local_means.append(_mean(entropies[start:end]))
    return max(0.0, _mean(local_means) - global_entropy_mean)


def _local_instability(
    *,
    indices: list[int],
    entropies: list[float],
    margins: list[float],
) -> float:
    if not indices:
        return 0.0
    local_scores = []
    for index in indices:
        if index >= len(entropies) or index >= len(margins):
            continue
        local_scores.append(entropies[index] - margins[index])
    return _variance(local_scores)


def _entity_region_scores(
    *,
    indices: list[int],
    entropies: list[float],
    margins: list[float],
) -> list[float]:
    if not indices:
        return []
    segment_size = max(1, len(entropies) // 3)
    scores: list[float] = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_indices = [index for index in indices if start <= index < end]
        if not segment_indices:
            continue
        scores.append(_subset_mean(entropies, segment_indices) - _subset_mean(margins, segment_indices))
    return scores


def _local_anomaly_peak(entropies: list[float], margins: list[float]) -> float:
    if not entropies or not margins:
        return 0.0
    peaks = []
    for entropy_mean, margin_mean in zip(_sliding_window_scores(entropies), _sliding_window_scores(margins)):
        peaks.append(entropy_mean - margin_mean)
    return max(peaks) if peaks else 0.0


def _longest_local_anomaly_run(entropies: list[float], margins: list[float]) -> int:
    longest = 0
    current = 0
    entropy_mean = _mean(entropies)
    margin_mean = _mean(margins)
    for entropy, margin in zip(entropies, margins):
        if entropy > entropy_mean + 0.2 or margin < margin_mean - 0.2:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _local_uncertainty_spike(entropies: list[float], global_entropy_mean: float) -> float:
    if not entropies:
        return 0.0
    return max(0.0, max(entropies) - global_entropy_mean)


def _local_margin_dip_contrast(margins: list[float], global_margin_mean: float) -> float:
    if not margins:
        return 0.0
    return max(0.0, global_margin_mean - min(margins))


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
