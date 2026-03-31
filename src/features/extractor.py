from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Protocol


FeatureMap = Mapping[str, float]


class FeatureExtractor(Protocol):
    def extract(
        self,
        prompt: str,
        response: str,
        token_stats: list["TokenUncertaintyStat"] | None = None,
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


class StructuralFeatureExtractor:
    def __init__(
        self,
        enable_uncertainty_proxies: bool = False,
        enable_token_uncertainty: bool = False,
    ) -> None:
        self.enable_uncertainty_proxies = enable_uncertainty_proxies
        self.enable_token_uncertainty = enable_token_uncertainty

    def extract(
        self,
        prompt: str,
        response: str,
        token_stats: list[TokenUncertaintyStat] | None = None,
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
            features.update(self._extract_token_uncertainty_features(token_stats))
        return features

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
            "token_mean_logprob": float(sum(logprobs) / len(logprobs)),
            "token_min_logprob": float(min(logprobs)),
            "token_entropy_mean": float(sum(entropies) / len(entropies)),
            "token_top1_top2_margin_mean": float(sum(margins) / len(margins)),
            "token_tail_low_confidence_rate": float(
                low_confidence_count / len(logprobs)
            ),
            "token_confidence_decay": float(early_mean - late_mean),
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]
