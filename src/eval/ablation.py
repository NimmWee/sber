from collections.abc import Mapping, Sequence

from features.extractor import StructuralFeatureExtractor, TokenUncertaintyStat


TOKEN_BASE_FEATURES = {
    "token_mean_logprob",
    "token_min_logprob",
    "token_entropy_mean",
    "token_top1_top2_margin_mean",
    "token_tail_low_confidence_rate",
    "token_confidence_decay",
}

TOKEN_GROUP_FEATURES = {
    "variance_std": {
        "token_logprob_variance",
        "token_logprob_std",
        "token_entropy_variance",
        "token_entropy_std",
        "token_top1_top2_margin_variance",
        "token_top1_top2_margin_std",
    },
    "segment_summaries": {
        "token_first_segment_mean_logprob",
        "token_middle_segment_mean_logprob",
        "token_last_segment_mean_logprob",
        "token_first_segment_mean_entropy",
        "token_middle_segment_mean_entropy",
        "token_last_segment_mean_entropy",
        "token_first_segment_mean_margin",
        "token_middle_segment_mean_margin",
        "token_last_segment_mean_margin",
    },
    "span_tail_rates": {
        "token_longest_low_confidence_span_length",
        "token_tail_low_confidence_rate_le_0_5",
        "token_tail_low_confidence_rate_le_1_0",
        "token_tail_low_confidence_rate_le_1_5",
    },
    "specialized_tokens": {
        "token_number_logprob_mean",
        "token_number_entropy_mean",
        "token_date_logprob_mean",
        "token_date_entropy_mean",
        "token_entity_like_logprob_mean",
        "token_entity_like_entropy_mean",
    },
}

FEATURE_GROUP_ORDER = (
    "base_token_uncertainty",
    "variance_std",
    "segment_summaries",
    "span_tail_rates",
    "specialized_tokens",
)


def extract_token_feature_groups(
    token_stats: list[TokenUncertaintyStat],
) -> dict[str, dict[str, float]]:
    groups = {
        "base_token_uncertainty": StructuralFeatureExtractor._extract_token_uncertainty_base(
            token_stats
        ),
        "variance_std": StructuralFeatureExtractor._extract_token_uncertainty_variance_std(
            token_stats
        ),
        "segment_summaries": StructuralFeatureExtractor._extract_token_uncertainty_segment_summaries(
            token_stats
        ),
        "span_tail_rates": StructuralFeatureExtractor._extract_token_uncertainty_span_tail_rates(
            token_stats
        ),
        "specialized_tokens": StructuralFeatureExtractor._extract_token_uncertainty_specialized(
            token_stats
        ),
    }
    return groups


def build_feature_allowlist(
    *,
    structural_feature_names: set[str],
    enabled_groups: Sequence[str],
) -> set[str]:
    allowlist = set(structural_feature_names)
    for group_name in enabled_groups:
        if group_name == "base_token_uncertainty":
            allowlist.update(TOKEN_BASE_FEATURES)
        else:
            allowlist.update(TOKEN_GROUP_FEATURES[group_name])
    return allowlist


def filter_feature_rows(
    *,
    feature_rows: list[Mapping[str, float]],
    allowed_features: set[str],
) -> list[dict[str, float]]:
    return [
        {
            feature_name: float(feature_value)
            for feature_name, feature_value in feature_row.items()
            if feature_name in allowed_features
        }
        for feature_row in feature_rows
    ]


def recommend_feature_group(*, pr_auc_delta: float, latency_delta_ms: float) -> str:
    if pr_auc_delta > 0.005:
        return "keep"
    if pr_auc_delta <= 0.0 and latency_delta_ms > 0.0:
        return "drop"
    return "keep"
