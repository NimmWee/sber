from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.default_detector import (
    DEFAULT_HEAD_KIND,
    DEFAULT_TOKEN_FEATURE_GROUPS,
    build_default_detector_extractor,
    build_default_detector_feature_allowlist,
    filter_default_detector_rows,
    train_default_detector_head,
)
from features.extractor import InternalModelSignal, TokenUncertaintyStat


def test_default_detector_extractor_enables_base_token_and_compact_internal_path() -> None:
    extractor = build_default_detector_extractor()

    assert extractor.enable_token_uncertainty is True
    assert extractor.enable_uncertainty_proxies is False
    assert extractor.enable_internal_features is True
    assert extractor.enable_compact_internal_enhancements is True

    features = extractor.extract(
        prompt="Who wrote Hamlet?",
        response="William Shakespeare wrote Hamlet in 1603.",
        token_stats=[
            TokenUncertaintyStat("William", -0.05, 0.12, 0.75),
            TokenUncertaintyStat("Shakespeare", -0.12, 0.18, 0.63),
            TokenUncertaintyStat("1603", -0.85, 0.55, 0.20),
        ],
        internal_signal=InternalModelSignal(
            last_layer_pooled_l2=1.25,
            last_layer_pooled_mean_abs=0.42,
            selected_layer_norm_variance=0.08,
            layer_disagreement_mean=0.12,
            selected_layer_disagreement_max=0.18,
            early_late_layer_consistency=0.84,
        ),
    )

    assert "token_mean_logprob" in features
    assert "token_confidence_decay" in features
    assert "token_logprob_std" not in features
    assert "internal_last_layer_pooled_l2" in features
    assert "internal_selected_layer_disagreement_max" in features
    assert "internal_entropy_disagreement_gap" in features


def test_default_detector_feature_allowlist_keeps_structural_base_token_and_internal_features() -> None:
    allowlist = build_default_detector_feature_allowlist(
        feature_rows=[
            {
                "response_length": 10.0,
                "token_mean_logprob": -0.2,
                "token_logprob_std": 0.3,
                "token_first_segment_mean_logprob": -0.1,
                "internal_last_layer_pooled_l2": 1.2,
                "internal_selected_layer_disagreement_max": 0.2,
            }
        ]
    )

    assert DEFAULT_TOKEN_FEATURE_GROUPS == ("base_token_uncertainty",)
    assert DEFAULT_HEAD_KIND == "lightgbm"
    assert "response_length" in allowlist
    assert "token_mean_logprob" in allowlist
    assert "token_confidence_decay" in allowlist
    assert "token_logprob_std" not in allowlist
    assert "token_first_segment_mean_logprob" not in allowlist
    assert "internal_last_layer_pooled_l2" in allowlist
    assert "internal_selected_layer_disagreement_max" in allowlist


def test_filter_default_detector_rows_drops_non_default_token_features_but_keeps_internal_features() -> None:
    filtered_rows = filter_default_detector_rows(
        feature_rows=[
            {
                "response_length": 10.0,
                "token_mean_logprob": -0.2,
                "token_logprob_std": 0.3,
                "internal_last_layer_pooled_l2": 1.2,
                "internal_selected_layer_disagreement_max": 0.2,
            }
        ]
    )

    assert filtered_rows == [
        {
            "response_length": 10.0,
            "token_mean_logprob": -0.2,
            "internal_last_layer_pooled_l2": 1.2,
            "internal_selected_layer_disagreement_max": 0.2,
        }
    ]


def test_train_default_detector_head_uses_lightgbm_shape() -> None:
    head = train_default_detector_head(
        feature_rows=[
            {
                "response_length": 10.0,
                "token_mean_logprob": -0.1,
                "internal_last_layer_pooled_l2": 1.0,
                "internal_selected_layer_disagreement_max": 0.1,
            },
            {
                "response_length": 12.0,
                "token_mean_logprob": -1.0,
                "internal_last_layer_pooled_l2": 1.4,
                "internal_selected_layer_disagreement_max": 0.4,
            },
        ],
        labels=[0, 1],
    )

    probabilities = head.predict_proba_batch(
        [
            {
                "response_length": 11.0,
                "token_mean_logprob": -0.4,
                "internal_last_layer_pooled_l2": 1.2,
                "internal_selected_layer_disagreement_max": 0.2,
            }
        ]
    )

    assert len(probabilities) == 1
    assert 0.0 <= probabilities[0] <= 1.0
