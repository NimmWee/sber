from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Protocol


FeatureMap = Mapping[str, float]


class FeatureExtractor(Protocol):
    def extract(
        self,
        prompt: str,
        response: str,
        token_stats: list["TokenUncertaintyStat"] | None = None,
        internal_signal: "InternalModelSignal" | None = None,
    ) -> FeatureMap:
        ...


TOKEN_PATTERN = re.compile(r"\w+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
SENTENCE_END_PATTERN = re.compile(r"[.!?]+")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


@dataclass(frozen=True)
class TokenUncertaintyStat:
    token: str
    logprob: float
    entropy: float
    top1_top2_margin: float


@dataclass(frozen=True)
class InternalModelSignal:
    last_layer_pooled_l2: float
    last_layer_pooled_mean_abs: float
    selected_layer_norm_variance: float
    layer_disagreement_mean: float


class StructuralFeatureExtractor:
    def __init__(
        self,
        enable_uncertainty_proxies: bool = False,
        enable_token_uncertainty: bool = False,
        enable_internal_features: bool = False,
        token_feature_groups: tuple[str, ...] | None = None,
    ) -> None:
        self.enable_uncertainty_proxies = enable_uncertainty_proxies
        self.enable_token_uncertainty = enable_token_uncertainty
        self.enable_internal_features = enable_internal_features
        self.token_feature_groups = token_feature_groups

    def extract(
        self,
        prompt: str,
        response: str,
        token_stats: list[TokenUncertaintyStat] | None = None,
        internal_signal: InternalModelSignal | None = None,
    ) -> FeatureMap:
        prompt_tokens = self._tokenize(prompt)
        response_tokens = self._tokenize(response)
        response_raw_tokens = TOKEN_PATTERN.findall(response)

        response_token_set = set(response_tokens)
        prompt_token_set = set(prompt_tokens)
        shared_token_count = len(response_token_set & prompt_token_set)
        response_unique_count = len(response_token_set)

        overlap_ratio = (
            shared_token_count / response_unique_count if response_unique_count else 0.0
        )
        novelty_ratio = (
            (response_unique_count - shared_token_count) / response_unique_count
            if response_unique_count
            else 0.0
        )

        sentence_count = len(SENTENCE_END_PATTERN.findall(response))
        if response.strip() and sentence_count == 0:
            sentence_count = 1

        features = {
            "response_length": float(len(response)),
            "token_count_proxy": float(len(response_tokens)),
            "digit_count": float(sum(character.isdigit() for character in response)),
            "punctuation_count": float(len(PUNCTUATION_PATTERN.findall(response))),
            "sentence_count_proxy": float(sentence_count),
            "prompt_response_overlap": float(overlap_ratio),
            "novelty_ratio_proxy": float(novelty_ratio),
        }
        if self.enable_uncertainty_proxies:
            features.update(
                self._extract_uncertainty_proxies(
                    prompt_token_set=prompt_token_set,
                    response_tokens=response_tokens,
                    response_raw_tokens=response_raw_tokens,
                    response=response,
                )
            )
        if self.enable_token_uncertainty and token_stats is not None:
            features.update(
                self._extract_token_uncertainty_features(
                    token_stats,
                    enabled_groups=self.token_feature_groups,
                )
            )
        if self.enable_internal_features and internal_signal is not None:
            features.update(self._extract_internal_features(internal_signal))
        return features

    @staticmethod
    def _extract_internal_features(
        internal_signal: InternalModelSignal,
    ) -> dict[str, float]:
        return {
            "internal_last_layer_pooled_l2": float(
                internal_signal.last_layer_pooled_l2
            ),
            "internal_last_layer_pooled_mean_abs": float(
                internal_signal.last_layer_pooled_mean_abs
            ),
            "internal_selected_layer_norm_variance": float(
                internal_signal.selected_layer_norm_variance
            ),
            "internal_layer_disagreement_mean": float(
                internal_signal.layer_disagreement_mean
            ),
        }

    def _extract_uncertainty_proxies(
        self,
        *,
        prompt_token_set: set[str],
        response_tokens: list[str],
        response_raw_tokens: list[str],
        response: str,
    ) -> dict[str, float]:
        token_count = len(response_tokens)
        if token_count == 0:
            return {
                "numeric_density": 0.0,
                "entity_like_token_density_proxy": 0.0,
                "prompt_response_lexical_divergence": 0.0,
                "late_position_numeric_density": 0.0,
                "date_density_proxy": 0.0,
            }

        numeric_token_count = sum(
            any(character.isdigit() for character in token) for token in response_raw_tokens
        )
        entity_like_count = sum(
            token[:1].isupper() and any(character.isalpha() for character in token)
            for token in response_raw_tokens
        )

        later_half_tokens = response_raw_tokens[token_count // 2 :]
        late_numeric_count = sum(
            any(character.isdigit() for character in token) for token in later_half_tokens
        )

        response_token_set = set(response_tokens)
        union_count = len(response_token_set | prompt_token_set)
        shared_count = len(response_token_set & prompt_token_set)
        lexical_divergence = (
            (union_count - shared_count) / union_count if union_count else 0.0
        )

        date_match_count = len(DATE_PATTERN.findall(response))

        return {
            "numeric_density": float(numeric_token_count / token_count),
            "entity_like_token_density_proxy": float(entity_like_count / token_count),
            "prompt_response_lexical_divergence": float(lexical_divergence),
            "late_position_numeric_density": float(
                late_numeric_count / len(later_half_tokens) if later_half_tokens else 0.0
            ),
            "date_density_proxy": float(date_match_count / token_count),
        }

    @staticmethod
    def _extract_token_uncertainty_features(
        token_stats: list[TokenUncertaintyStat],
        *,
        enabled_groups: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        group_order = enabled_groups or (
            "base_token_uncertainty",
            "variance_std",
            "segment_summaries",
            "span_tail_rates",
            "specialized_tokens",
        )
        features: dict[str, float] = {}
        for group_name in group_order:
            if group_name == "base_token_uncertainty":
                features.update(
                    StructuralFeatureExtractor._extract_token_uncertainty_base(
                        token_stats
                    )
                )
            elif group_name == "variance_std":
                features.update(
                    StructuralFeatureExtractor._extract_token_uncertainty_variance_std(
                        token_stats
                    )
                )
            elif group_name == "segment_summaries":
                features.update(
                    StructuralFeatureExtractor._extract_token_uncertainty_segment_summaries(
                        token_stats
                    )
                )
            elif group_name == "span_tail_rates":
                features.update(
                    StructuralFeatureExtractor._extract_token_uncertainty_span_tail_rates(
                        token_stats
                    )
                )
            elif group_name == "specialized_tokens":
                features.update(
                    StructuralFeatureExtractor._extract_token_uncertainty_specialized(
                        token_stats
                    )
                )
        return features

    @staticmethod
    def _extract_token_uncertainty_base(
        token_stats: list[TokenUncertaintyStat],
    ) -> dict[str, float]:
        if not token_stats:
            return {
                "token_mean_logprob": 0.0,
                "token_min_logprob": 0.0,
                "token_entropy_mean": 0.0,
                "token_top1_top2_margin_mean": 0.0,
                "token_tail_low_confidence_rate": 0.0,
                "token_confidence_decay": 0.0,
            }

        logprobs = [float(stat.logprob) for stat in token_stats]
        entropies = [float(stat.entropy) for stat in token_stats]
        margins = [float(stat.top1_top2_margin) for stat in token_stats]
        low_confidence_count = sum(logprob <= -1.0 for logprob in logprobs)

        midpoint = len(logprobs) // 2
        early_slice = logprobs[:midpoint] or logprobs
        late_slice = logprobs[midpoint:] or logprobs
        early_mean = sum(early_slice) / len(early_slice)
        late_mean = sum(late_slice) / len(late_slice)

        return {
            "token_mean_logprob": float(StructuralFeatureExtractor._mean(logprobs)),
            "token_min_logprob": float(min(logprobs)),
            "token_entropy_mean": float(StructuralFeatureExtractor._mean(entropies)),
            "token_top1_top2_margin_mean": float(
                StructuralFeatureExtractor._mean(margins)
            ),
            "token_tail_low_confidence_rate": float(
                low_confidence_count / len(logprobs)
            ),
            "token_confidence_decay": float(early_mean - late_mean),
        }

    @staticmethod
    def _extract_token_uncertainty_variance_std(
        token_stats: list[TokenUncertaintyStat],
    ) -> dict[str, float]:
        if not token_stats:
            return {
                "token_logprob_variance": 0.0,
                "token_logprob_std": 0.0,
                "token_entropy_variance": 0.0,
                "token_entropy_std": 0.0,
                "token_top1_top2_margin_variance": 0.0,
                "token_top1_top2_margin_std": 0.0,
            }

        logprobs = [float(stat.logprob) for stat in token_stats]
        entropies = [float(stat.entropy) for stat in token_stats]
        margins = [float(stat.top1_top2_margin) for stat in token_stats]

        return {
            "token_logprob_variance": float(
                StructuralFeatureExtractor._variance(logprobs)
            ),
            "token_logprob_std": float(StructuralFeatureExtractor._std(logprobs)),
            "token_entropy_variance": float(
                StructuralFeatureExtractor._variance(entropies)
            ),
            "token_entropy_std": float(StructuralFeatureExtractor._std(entropies)),
            "token_top1_top2_margin_variance": float(
                StructuralFeatureExtractor._variance(margins)
            ),
            "token_top1_top2_margin_std": float(
                StructuralFeatureExtractor._std(margins)
            ),
        }

    @staticmethod
    def _extract_token_uncertainty_segment_summaries(
        token_stats: list[TokenUncertaintyStat],
    ) -> dict[str, float]:
        if not token_stats:
            return {
                "token_first_segment_mean_logprob": 0.0,
                "token_middle_segment_mean_logprob": 0.0,
                "token_last_segment_mean_logprob": 0.0,
                "token_first_segment_mean_entropy": 0.0,
                "token_middle_segment_mean_entropy": 0.0,
                "token_last_segment_mean_entropy": 0.0,
                "token_first_segment_mean_margin": 0.0,
                "token_middle_segment_mean_margin": 0.0,
                "token_last_segment_mean_margin": 0.0,
            }

        logprobs = [float(stat.logprob) for stat in token_stats]
        entropies = [float(stat.entropy) for stat in token_stats]
        margins = [float(stat.top1_top2_margin) for stat in token_stats]

        return {
            "token_first_segment_mean_logprob": float(
                StructuralFeatureExtractor._segment_mean(logprobs, "first")
            ),
            "token_middle_segment_mean_logprob": float(
                StructuralFeatureExtractor._segment_mean(logprobs, "middle")
            ),
            "token_last_segment_mean_logprob": float(
                StructuralFeatureExtractor._segment_mean(logprobs, "last")
            ),
            "token_first_segment_mean_entropy": float(
                StructuralFeatureExtractor._segment_mean(entropies, "first")
            ),
            "token_middle_segment_mean_entropy": float(
                StructuralFeatureExtractor._segment_mean(entropies, "middle")
            ),
            "token_last_segment_mean_entropy": float(
                StructuralFeatureExtractor._segment_mean(entropies, "last")
            ),
            "token_first_segment_mean_margin": float(
                StructuralFeatureExtractor._segment_mean(margins, "first")
            ),
            "token_middle_segment_mean_margin": float(
                StructuralFeatureExtractor._segment_mean(margins, "middle")
            ),
            "token_last_segment_mean_margin": float(
                StructuralFeatureExtractor._segment_mean(margins, "last")
            ),
        }

    @staticmethod
    def _extract_token_uncertainty_span_tail_rates(
        token_stats: list[TokenUncertaintyStat],
    ) -> dict[str, float]:
        if not token_stats:
            return {
                "token_longest_low_confidence_span_length": 0.0,
                "token_tail_low_confidence_rate_le_0_5": 0.0,
                "token_tail_low_confidence_rate_le_1_0": 0.0,
                "token_tail_low_confidence_rate_le_1_5": 0.0,
            }

        logprobs = [float(stat.logprob) for stat in token_stats]
        tail_logprobs = StructuralFeatureExtractor._tail_slice(logprobs)

        return {
            "token_longest_low_confidence_span_length": float(
                StructuralFeatureExtractor._longest_low_confidence_span(
                    logprobs,
                    threshold=-1.0,
                )
            ),
            "token_tail_low_confidence_rate_le_0_5": float(
                StructuralFeatureExtractor._low_confidence_rate(
                    tail_logprobs,
                    threshold=-0.5,
                )
            ),
            "token_tail_low_confidence_rate_le_1_0": float(
                StructuralFeatureExtractor._low_confidence_rate(
                    tail_logprobs,
                    threshold=-1.0,
                )
            ),
            "token_tail_low_confidence_rate_le_1_5": float(
                StructuralFeatureExtractor._low_confidence_rate(
                    tail_logprobs,
                    threshold=-1.5,
                )
            ),
        }

    @staticmethod
    def _extract_token_uncertainty_specialized(
        token_stats: list[TokenUncertaintyStat],
    ) -> dict[str, float]:
        if not token_stats:
            return {
                "token_number_logprob_mean": 0.0,
                "token_number_entropy_mean": 0.0,
                "token_date_logprob_mean": 0.0,
                "token_date_entropy_mean": 0.0,
                "token_entity_like_logprob_mean": 0.0,
                "token_entity_like_entropy_mean": 0.0,
            }

        number_stats = [
            stat
            for stat in token_stats
            if StructuralFeatureExtractor._is_number_like(stat.token)
        ]
        date_stats = [
            stat for stat in token_stats if StructuralFeatureExtractor._is_date_like(stat.token)
        ]
        entity_stats = [
            stat
            for stat in token_stats
            if StructuralFeatureExtractor._is_entity_like(stat.token)
        ]

        return {
            "token_number_logprob_mean": float(
                StructuralFeatureExtractor._subset_mean(number_stats, "logprob")
            ),
            "token_number_entropy_mean": float(
                StructuralFeatureExtractor._subset_mean(number_stats, "entropy")
            ),
            "token_date_logprob_mean": float(
                StructuralFeatureExtractor._subset_mean(date_stats, "logprob")
            ),
            "token_date_entropy_mean": float(
                StructuralFeatureExtractor._subset_mean(date_stats, "entropy")
            ),
            "token_entity_like_logprob_mean": float(
                StructuralFeatureExtractor._subset_mean(entity_stats, "logprob")
            ),
            "token_entity_like_entropy_mean": float(
                StructuralFeatureExtractor._subset_mean(entity_stats, "entropy")
            ),
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _variance(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = StructuralFeatureExtractor._mean(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    @staticmethod
    def _std(values: list[float]) -> float:
        return math.sqrt(StructuralFeatureExtractor._variance(values))

    @staticmethod
    def _segment_mean(values: list[float], segment: str) -> float:
        segment_values = StructuralFeatureExtractor._segment_slice(values, segment)
        return StructuralFeatureExtractor._mean(segment_values)

    @staticmethod
    def _segment_slice(values: list[float], segment: str) -> list[float]:
        if not values:
            return []

        length = len(values)
        third = max(1, length // 3)
        if segment == "first":
            selected = values[:third]
        elif segment == "middle":
            start = max(0, (length - third) // 2)
            selected = values[start : start + third]
        else:
            selected = values[-third:]
        return selected or values

    @staticmethod
    def _tail_slice(values: list[float]) -> list[float]:
        if not values:
            return []
        tail_size = max(1, len(values) // 3)
        return values[-tail_size:]

    @staticmethod
    def _low_confidence_rate(values: list[float], threshold: float) -> float:
        if not values:
            return 0.0
        return sum(value <= threshold for value in values) / len(values)

    @staticmethod
    def _longest_low_confidence_span(values: list[float], threshold: float) -> int:
        longest = 0
        current = 0
        for value in values:
            if value <= threshold:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    @staticmethod
    def _subset_mean(
        token_stats: list[TokenUncertaintyStat],
        attribute: str,
    ) -> float:
        if not token_stats:
            return 0.0
        values = [float(getattr(stat, attribute)) for stat in token_stats]
        return StructuralFeatureExtractor._mean(values)

    @staticmethod
    def _normalized_token_text(token: str) -> str:
        return token.lstrip(" Ġ▁")

    @staticmethod
    def _is_number_like(token: str) -> bool:
        normalized = StructuralFeatureExtractor._normalized_token_text(token)
        return any(character.isdigit() for character in normalized)

    @staticmethod
    def _is_date_like(token: str) -> bool:
        normalized = StructuralFeatureExtractor._normalized_token_text(token)
        return bool(DATE_PATTERN.search(normalized))

    @staticmethod
    def _is_entity_like(token: str) -> bool:
        normalized = StructuralFeatureExtractor._normalized_token_text(token)
        return normalized[:1].isupper() and any(
            character.isalpha() for character in normalized
        )
