from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.default_detector import (
    DEFAULT_TOKEN_FEATURE_GROUPS,
    build_default_detector_extractor,
    build_default_detector_feature_allowlist,
    filter_default_detector_rows,
)
from features.extractor import TokenUncertaintyStat


def test_default_detector_extractor_enables_only_base_token_path() -> None:
    extractor = build_default_detector_extractor()

    assert extractor.enable_token_uncertainty is True
    assert extractor.enable_uncertainty_proxies is False
    assert extractor.enable_internal_features is False

    features = extractor.extract(
        prompt="Who wrote Hamlet?",
        response="William Shakespeare wrote Hamlet in 1603.",
        token_stats=[
            TokenUncertaintyStat("William", -0.05, 0.12, 0.75),
            TokenUncertaintyStat("Shakespeare", -0.12, 0.18, 0.63),
            TokenUncertaintyStat("1603", -0.85, 0.55, 0.20),
        ],
    )

    assert "token_mean_logprob" in features
    assert "token_confidence_decay" in features
    assert "token_logprob_std" not in features
    assert "token_first_segment_mean_logprob" not in features


def test_default_detector_feature_allowlist_keeps_only_structural_and_base_token_features() -> None:
    allowlist = build_default_detector_feature_allowlist(
        feature_rows=[
            {
                "response_length": 10.0,
                "token_mean_logprob": -0.2,
                "token_logprob_std": 0.3,
                "token_first_segment_mean_logprob": -0.1,
                "internal_last_layer_pooled_l2": 1.2,
            }
        ]
    )

    assert DEFAULT_TOKEN_FEATURE_GROUPS == ("base_token_uncertainty",)
    assert "response_length" in allowlist
    assert "token_mean_logprob" in allowlist
    assert "token_confidence_decay" in allowlist
    assert "token_logprob_std" not in allowlist
    assert "token_first_segment_mean_logprob" not in allowlist
    assert "internal_last_layer_pooled_l2" not in allowlist


def test_filter_default_detector_rows_drops_non_default_token_and_internal_features() -> None:
    filtered_rows = filter_default_detector_rows(
        feature_rows=[
            {
                "response_length": 10.0,
                "token_mean_logprob": -0.2,
                "token_logprob_std": 0.3,
                "internal_last_layer_pooled_l2": 1.2,
            }
        ]
    )

    assert filtered_rows == [
        {
            "response_length": 10.0,
            "token_mean_logprob": -0.2,
        }
    ]
