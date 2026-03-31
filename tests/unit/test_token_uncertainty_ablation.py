from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.ablation import (
    FEATURE_GROUP_ORDER,
    TOKEN_BASE_FEATURES,
    TOKEN_GROUP_FEATURES,
    build_feature_allowlist,
    extract_token_feature_groups,
    filter_feature_rows,
    recommend_feature_group,
)
from features.extractor import TokenUncertaintyStat


def _token_stats() -> list[TokenUncertaintyStat]:
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


def test_extract_token_feature_groups_returns_expected_group_keys() -> None:
    groups = extract_token_feature_groups(_token_stats())

    assert set(groups.keys()) == set(FEATURE_GROUP_ORDER)
    assert TOKEN_BASE_FEATURES.issubset(groups["base_token_uncertainty"].keys())
    assert TOKEN_GROUP_FEATURES["variance_std"].issubset(groups["variance_std"].keys())
    assert TOKEN_GROUP_FEATURES["segment_summaries"].issubset(
        groups["segment_summaries"].keys()
    )


def test_build_feature_allowlist_unions_requested_groups() -> None:
    allowlist = build_feature_allowlist(
        structural_feature_names={"response_length", "token_count_proxy"},
        enabled_groups=("base_token_uncertainty", "variance_std"),
    )

    assert "response_length" in allowlist
    assert "token_mean_logprob" in allowlist
    assert "token_logprob_std" in allowlist
    assert "token_first_segment_mean_logprob" not in allowlist


def test_filter_feature_rows_keeps_only_allowed_features() -> None:
    rows = [
        {
            "response_length": 10.0,
            "token_mean_logprob": -0.3,
            "token_logprob_std": 0.4,
            "token_first_segment_mean_logprob": -0.2,
        }
    ]

    filtered = filter_feature_rows(
        feature_rows=rows,
        allowed_features={"response_length", "token_mean_logprob"},
    )

    assert filtered == [{"response_length": 10.0, "token_mean_logprob": -0.3}]


def test_recommend_feature_group_prefers_quality_gain_with_small_latency_cost() -> None:
    recommendation = recommend_feature_group(pr_auc_delta=0.02, latency_delta_ms=0.01)

    assert recommendation == "keep"


def test_recommend_feature_group_drops_no_gain_with_extra_latency() -> None:
    recommendation = recommend_feature_group(pr_auc_delta=0.0, latency_delta_ms=0.05)

    assert recommendation == "drop"
