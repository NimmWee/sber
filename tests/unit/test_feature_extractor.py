from collections.abc import Mapping
from dataclasses import replace
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from features.extractor import (
    InternalModelSignal,
    StructuralFeatureExtractor,
    TokenUncertaintyStat,
)


EXPECTED_FEATURES = {
    "response_length",
    "token_count_proxy",
    "digit_count",
    "punctuation_count",
    "sentence_count_proxy",
    "prompt_response_overlap",
    "novelty_ratio_proxy",
}

UNCERTAINTY_FEATURES = {
    "numeric_density",
    "entity_like_token_density_proxy",
    "prompt_response_lexical_divergence",
    "late_position_numeric_density",
    "date_density_proxy",
}

TOKEN_UNCERTAINTY_FEATURES = {
    "token_mean_logprob",
    "token_min_logprob",
    "token_logprob_variance",
    "token_logprob_std",
    "token_entropy_mean",
    "token_entropy_variance",
    "token_entropy_std",
    "token_top1_top2_margin_mean",
    "token_top1_top2_margin_variance",
    "token_top1_top2_margin_std",
    "token_tail_low_confidence_rate",
    "token_confidence_decay",
    "token_first_segment_mean_logprob",
    "token_middle_segment_mean_logprob",
    "token_last_segment_mean_logprob",
    "token_first_segment_mean_entropy",
    "token_middle_segment_mean_entropy",
    "token_last_segment_mean_entropy",
    "token_first_segment_mean_margin",
    "token_middle_segment_mean_margin",
    "token_last_segment_mean_margin",
    "token_longest_low_confidence_span_length",
    "token_tail_low_confidence_rate_le_0_5",
    "token_tail_low_confidence_rate_le_1_0",
    "token_tail_low_confidence_rate_le_1_5",
    "token_number_logprob_mean",
    "token_number_entropy_mean",
    "token_date_logprob_mean",
    "token_date_entropy_mean",
    "token_entity_like_logprob_mean",
    "token_entity_like_entropy_mean",
}

INTERNAL_FEATURES = {
    "internal_last_layer_pooled_l2",
    "internal_last_layer_pooled_mean_abs",
    "internal_selected_layer_norm_variance",
    "internal_layer_disagreement_mean",
}


def _token_stats() -> list[TokenUncertaintyStat]:
    return [
        TokenUncertaintyStat(
            token="William",
            logprob=-0.05,
            entropy=0.12,
            top1_top2_margin=0.75,
        ),
        TokenUncertaintyStat(
            token="Shakespeare",
            logprob=-0.12,
            entropy=0.18,
            top1_top2_margin=0.63,
        ),
        TokenUncertaintyStat(
            token="1603",
            logprob=-0.85,
            entropy=0.55,
            top1_top2_margin=0.20,
        ),
    ]


def _extended_token_stats() -> list[TokenUncertaintyStat]:
    return [
        TokenUncertaintyStat("Alice", -0.08, 0.18, 0.70),
        TokenUncertaintyStat("visited", -0.12, 0.22, 0.62),
        TokenUncertaintyStat("Paris", -0.35, 0.40, 0.38),
        TokenUncertaintyStat("on", -0.18, 0.21, 0.58),
        TokenUncertaintyStat("2024-01-15", -1.30, 0.95, 0.12),
        TokenUncertaintyStat("with", -0.22, 0.24, 0.52),
        TokenUncertaintyStat("3", -1.45, 1.05, 0.08),
        TokenUncertaintyStat("companions", -0.28, 0.30, 0.44),
        TokenUncertaintyStat("yesterday", -1.10, 0.90, 0.15),
    ]


def test_extract_returns_mapping_of_float_features() -> None:
    extractor = StructuralFeatureExtractor()

    features = extractor.extract(
        prompt="What is the capital of France?",
        response="Paris is the capital of France.",
    )

    assert isinstance(features, Mapping)
    assert EXPECTED_FEATURES.issubset(features.keys())
    assert all(isinstance(value, float) for value in features.values())


def test_extract_is_deterministic_for_same_input() -> None:
    extractor = StructuralFeatureExtractor()
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare wrote Hamlet."

    first = extractor.extract(prompt=prompt, response=response)
    second = extractor.extract(prompt=prompt, response=response)

    assert dict(first) == dict(second)


def test_extract_never_returns_nan_or_inf_values() -> None:
    extractor = StructuralFeatureExtractor()

    features = extractor.extract(
        prompt="Summarize the treaty.",
        response="The treaty was signed on 2024-03-01 and revised in 2025.",
    )

    assert all(math.isfinite(value) for value in features.values())


def test_extract_handles_empty_response() -> None:
    extractor = StructuralFeatureExtractor()

    features = extractor.extract(prompt="Answer briefly.", response="")

    assert features["response_length"] == 0.0
    assert features["token_count_proxy"] == 0.0
    assert features["novelty_ratio_proxy"] == 0.0


def test_extract_handles_long_response() -> None:
    extractor = StructuralFeatureExtractor()
    response = "fact " * 1000

    features = extractor.extract(prompt="Explain the evidence.", response=response)

    assert features["response_length"] > 0.0
    assert features["token_count_proxy"] >= 1000.0


def test_extract_handles_numbers_and_dates() -> None:
    extractor = StructuralFeatureExtractor()
    response = "The mission launched on 2024-01-15 and returned 3 samples in 2025."

    features = extractor.extract(prompt="State mission details.", response=response)

    assert features["digit_count"] > 0.0
    assert features["sentence_count_proxy"] >= 1.0


def test_extract_with_uncertainty_proxies_returns_mapping_of_float_features() -> None:
    extractor = StructuralFeatureExtractor(enable_uncertainty_proxies=True)

    features = extractor.extract(
        prompt="What year was the treaty signed?",
        response="The treaty was signed in 2024.",
    )

    assert isinstance(features, Mapping)
    assert EXPECTED_FEATURES.issubset(features.keys())
    assert UNCERTAINTY_FEATURES.issubset(features.keys())
    assert all(isinstance(value, float) for value in features.values())


def test_extract_with_uncertainty_proxies_is_deterministic() -> None:
    extractor = StructuralFeatureExtractor(enable_uncertainty_proxies=True)
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare wrote Hamlet in 1603."

    first = extractor.extract(prompt=prompt, response=response)
    second = extractor.extract(prompt=prompt, response=response)

    assert dict(first) == dict(second)


def test_extract_with_uncertainty_proxies_never_returns_nan_or_inf() -> None:
    extractor = StructuralFeatureExtractor(enable_uncertainty_proxies=True)

    features = extractor.extract(
        prompt="Summarize the treaty revision.",
        response="The treaty was revised on 2024-03-01 and signed again in 2025.",
    )

    assert all(math.isfinite(value) for value in features.values())


def test_extract_with_uncertainty_proxies_handles_empty_response() -> None:
    extractor = StructuralFeatureExtractor(enable_uncertainty_proxies=True)

    features = extractor.extract(prompt="Answer briefly.", response="")

    assert features["numeric_density"] == 0.0
    assert features["entity_like_token_density_proxy"] == 0.0
    assert features["prompt_response_lexical_divergence"] == 0.0
    assert features["late_position_numeric_density"] == 0.0
    assert features["date_density_proxy"] == 0.0


def test_extract_with_uncertainty_proxies_handles_numbers_and_dates() -> None:
    extractor = StructuralFeatureExtractor(enable_uncertainty_proxies=True)
    response = "NASA launched Artemis on 2024-01-15 and recovered 3 capsules in 2025."

    features = extractor.extract(prompt="State mission details.", response=response)

    assert features["numeric_density"] > 0.0
    assert features["date_density_proxy"] > 0.0
    assert features["entity_like_token_density_proxy"] > 0.0


def test_new_features_are_only_present_when_enabled() -> None:
    baseline = StructuralFeatureExtractor(enable_uncertainty_proxies=False)
    enriched = StructuralFeatureExtractor(enable_uncertainty_proxies=True)
    prompt = "State mission details."
    response = "NASA launched Artemis on 2024-01-15 and recovered 3 capsules in 2025."

    baseline_features = baseline.extract(prompt=prompt, response=response)
    enriched_features = enriched.extract(prompt=prompt, response=response)

    assert UNCERTAINTY_FEATURES.isdisjoint(baseline_features.keys())
    assert UNCERTAINTY_FEATURES.issubset(enriched_features.keys())


def test_extract_with_token_uncertainty_returns_mapping_of_float_features() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    features = extractor.extract(
        prompt="Who wrote Hamlet?",
        response="William Shakespeare wrote Hamlet in 1603.",
        token_stats=_token_stats(),
    )

    assert isinstance(features, Mapping)
    assert EXPECTED_FEATURES.issubset(features.keys())
    assert TOKEN_UNCERTAINTY_FEATURES.issubset(features.keys())
    assert all(isinstance(value, float) for value in features.values())


def test_extract_with_token_uncertainty_is_deterministic() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare wrote Hamlet in 1603."
    token_stats = _token_stats()

    first = extractor.extract(prompt=prompt, response=response, token_stats=token_stats)
    second = extractor.extract(prompt=prompt, response=response, token_stats=token_stats)

    assert dict(first) == dict(second)


def test_extract_with_token_uncertainty_never_returns_nan_or_inf() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    features = extractor.extract(
        prompt="Summarize the treaty revision.",
        response="The treaty was revised on 2024-03-01 and signed again in 2025.",
        token_stats=_token_stats(),
    )

    assert all(math.isfinite(value) for value in features.values())


def test_extract_with_token_uncertainty_handles_empty_response() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    features = extractor.extract(prompt="Answer briefly.", response="", token_stats=[])

    assert features["token_mean_logprob"] == 0.0
    assert features["token_min_logprob"] == 0.0
    assert features["token_entropy_mean"] == 0.0
    assert features["token_top1_top2_margin_mean"] == 0.0
    assert features["token_tail_low_confidence_rate"] == 0.0
    assert features["token_confidence_decay"] == 0.0
    assert features["token_logprob_std"] == 0.0
    assert features["token_entropy_std"] == 0.0
    assert features["token_top1_top2_margin_std"] == 0.0
    assert features["token_longest_low_confidence_span_length"] == 0.0
    assert features["token_tail_low_confidence_rate_le_1_5"] == 0.0
    assert features["token_number_logprob_mean"] == 0.0
    assert features["token_date_entropy_mean"] == 0.0


def test_extract_with_token_uncertainty_handles_long_response() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)
    response = "fact " * 1000
    token_stats = [
        TokenUncertaintyStat(
            token=f"fact{index}",
            logprob=-0.1 - index * 0.0001,
            entropy=0.2,
            top1_top2_margin=0.5,
        )
        for index in range(1000)
    ]

    features = extractor.extract(
        prompt="Explain the evidence.",
        response=response,
        token_stats=token_stats,
    )

    assert features["token_mean_logprob"] < 0.0
    assert features["token_tail_low_confidence_rate"] >= 0.0


def test_extract_with_token_uncertainty_handles_numbers_and_dates() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)
    token_stats = [
        TokenUncertaintyStat(token="NASA", logprob=-0.1, entropy=0.2, top1_top2_margin=0.7),
        TokenUncertaintyStat(token="2024-01-15", logprob=-0.9, entropy=0.8, top1_top2_margin=0.1),
        TokenUncertaintyStat(token="3", logprob=-1.1, entropy=0.9, top1_top2_margin=0.08),
    ]

    features = extractor.extract(
        prompt="State mission details.",
        response="NASA launched Artemis on 2024-01-15 and recovered 3 capsules.",
        token_stats=token_stats,
    )

    assert features["token_min_logprob"] <= features["token_mean_logprob"]
    assert features["token_tail_low_confidence_rate"] > 0.0


def test_extract_with_token_uncertainty_exposes_extended_aggregation_features() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    features = extractor.extract(
        prompt="State the travel details.",
        response="Alice visited Paris on 2024-01-15 with 3 companions yesterday.",
        token_stats=_extended_token_stats(),
    )

    assert features["token_logprob_variance"] > 0.0
    assert features["token_entropy_variance"] > 0.0
    assert features["token_top1_top2_margin_variance"] > 0.0
    assert features["token_longest_low_confidence_span_length"] >= 1.0
    assert (
        features["token_tail_low_confidence_rate_le_0_5"]
        >= features["token_tail_low_confidence_rate_le_1_0"]
        >= features["token_tail_low_confidence_rate_le_1_5"]
    )
    assert features["token_number_logprob_mean"] < 0.0
    assert features["token_date_entropy_mean"] > 0.0
    assert features["token_entity_like_logprob_mean"] < 0.0


def test_extract_with_token_uncertainty_segment_summaries_are_finite_and_present() -> None:
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)

    features = extractor.extract(
        prompt="State the travel details.",
        response="Alice visited Paris on 2024-01-15 with 3 companions yesterday.",
        token_stats=_extended_token_stats(),
    )

    segment_feature_names = {
        "token_first_segment_mean_logprob",
        "token_middle_segment_mean_logprob",
        "token_last_segment_mean_logprob",
        "token_first_segment_mean_entropy",
        "token_middle_segment_mean_entropy",
        "token_last_segment_mean_entropy",
        "token_first_segment_mean_margin",
        "token_middle_segment_mean_margin",
        "token_last_segment_mean_margin",
    }

    assert segment_feature_names.issubset(features.keys())
    assert all(math.isfinite(features[name]) for name in segment_feature_names)


def test_token_uncertainty_features_are_only_present_when_enabled_and_provided() -> None:
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare wrote Hamlet in 1603."
    token_stats = _token_stats()

    disabled = StructuralFeatureExtractor(enable_token_uncertainty=False).extract(
        prompt=prompt,
        response=response,
        token_stats=token_stats,
    )
    enabled_without_stats = StructuralFeatureExtractor(
        enable_token_uncertainty=True
    ).extract(prompt=prompt, response=response)
    enabled_with_stats = StructuralFeatureExtractor(enable_token_uncertainty=True).extract(
        prompt=prompt,
        response=response,
        token_stats=token_stats,
    )

    assert TOKEN_UNCERTAINTY_FEATURES.isdisjoint(disabled.keys())
    assert TOKEN_UNCERTAINTY_FEATURES.isdisjoint(enabled_without_stats.keys())
    assert TOKEN_UNCERTAINTY_FEATURES.issubset(enabled_with_stats.keys())


def test_extract_with_internal_features_returns_mapping_of_float_features() -> None:
    extractor = StructuralFeatureExtractor(enable_internal_features=True)

    features = extractor.extract(
        prompt="Who wrote Hamlet?",
        response="William Shakespeare wrote Hamlet.",
        internal_signal=InternalModelSignal(
            last_layer_pooled_l2=1.25,
            last_layer_pooled_mean_abs=0.42,
            selected_layer_norm_variance=0.08,
            layer_disagreement_mean=0.12,
        ),
    )

    assert isinstance(features, Mapping)
    assert EXPECTED_FEATURES.issubset(features.keys())
    assert INTERNAL_FEATURES.issubset(features.keys())
    assert all(isinstance(value, float) for value in features.values())


def test_extract_with_internal_features_is_deterministic() -> None:
    extractor = StructuralFeatureExtractor(enable_internal_features=True)
    internal_signal = InternalModelSignal(
        last_layer_pooled_l2=1.25,
        last_layer_pooled_mean_abs=0.42,
        selected_layer_norm_variance=0.08,
        layer_disagreement_mean=0.12,
    )

    first = extractor.extract(
        prompt="What is the capital of Italy?",
        response="Rome is the capital of Italy.",
        internal_signal=internal_signal,
    )
    second = extractor.extract(
        prompt="What is the capital of Italy?",
        response="Rome is the capital of Italy.",
        internal_signal=internal_signal,
    )

    assert first == second


def test_extract_with_internal_features_never_returns_nan_or_inf() -> None:
    extractor = StructuralFeatureExtractor(enable_internal_features=True)

    features = extractor.extract(
        prompt="Summarize the treaty revision.",
        response="The treaty was revised on 2024-03-01 and signed again in 2025.",
        internal_signal=InternalModelSignal(
            last_layer_pooled_l2=1.25,
            last_layer_pooled_mean_abs=0.42,
            selected_layer_norm_variance=0.08,
            layer_disagreement_mean=0.12,
        ),
    )

    assert all(math.isfinite(value) for value in features.values())


def test_internal_features_are_only_present_when_enabled_and_provided() -> None:
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare wrote Hamlet."
    internal_signal = InternalModelSignal(
        last_layer_pooled_l2=1.25,
        last_layer_pooled_mean_abs=0.42,
        selected_layer_norm_variance=0.08,
        layer_disagreement_mean=0.12,
    )

    disabled = StructuralFeatureExtractor(enable_internal_features=False).extract(
        prompt=prompt,
        response=response,
        internal_signal=internal_signal,
    )
    enabled_without_signal = StructuralFeatureExtractor(
        enable_internal_features=True
    ).extract(prompt=prompt, response=response)
    enabled_with_signal = StructuralFeatureExtractor(enable_internal_features=True).extract(
        prompt=prompt,
        response=response,
        internal_signal=internal_signal,
    )

    assert INTERNAL_FEATURES.isdisjoint(disabled.keys())
    assert INTERNAL_FEATURES.isdisjoint(enabled_without_signal.keys())
    assert INTERNAL_FEATURES.issubset(enabled_with_signal.keys())
