from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.specialist_ensemble import extract_specialist_features, build_fusion_feature_row
from features.extractor import InternalModelSignal, TokenUncertaintyStat


def test_extract_specialist_features_returns_finite_prefixed_groups() -> None:
    token_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.3, 0.6),
        TokenUncertaintyStat("2024", -1.1, 0.8, 0.1),
        TokenUncertaintyStat("Berlin", -0.9, 0.7, 0.15),
        TokenUncertaintyStat("42", -1.0, 0.9, 0.08),
    ]
    internal_signal = InternalModelSignal(
        last_layer_pooled_l2=1.1,
        last_layer_pooled_mean_abs=0.4,
        selected_layer_norm_variance=0.07,
        layer_disagreement_mean=0.12,
        selected_layer_disagreement_max=0.18,
        early_late_layer_consistency=0.81,
    )

    features = extract_specialist_features(
        prompt="Which city hosted the 2024 summit?",
        response="Paris hosted the 2024 summit while Berlin sent 42 delegates.",
        token_stats=token_stats,
        internal_signal=internal_signal,
    )

    assert any(name.startswith("specialist_numeric_") for name in features)
    assert any(name.startswith("specialist_entity_") for name in features)
    assert any(name.startswith("specialist_long_") for name in features)
    assert all(isinstance(value, float) for value in features.values())
    assert all(value == value for value in features.values())


def test_build_fusion_feature_row_includes_baseline_and_specialist_scores() -> None:
    row = build_fusion_feature_row(
        baseline_score=0.41,
        numeric_score=0.62,
        entity_score=0.55,
        long_score=0.48,
        specialist_features={
            "specialist_numeric_density": 0.3,
            "specialist_entity_density": 0.4,
            "specialist_long_length_bucket": 1.0,
        },
    )

    assert row["baseline_score"] == 0.41
    assert row["numeric_specialist_score"] == 0.62
    assert row["entity_specialist_score"] == 0.55
    assert row["long_specialist_score"] == 0.48
    assert row["specialist_numeric_density"] == 0.3


def test_extract_specialist_features_adds_local_numeric_entity_and_long_anomaly_signals() -> None:
    token_stats = [
        TokenUncertaintyStat("In", -0.1, 0.2, 0.7),
        TokenUncertaintyStat("2024", -1.4, 1.1, 0.08),
        TokenUncertaintyStat("Paris", -1.0, 0.9, 0.12),
        TokenUncertaintyStat("reported", -0.2, 0.3, 0.65),
        TokenUncertaintyStat("43", -1.3, 1.0, 0.09),
        TokenUncertaintyStat("delegates", -0.3, 0.4, 0.55),
        TokenUncertaintyStat("Berlin", -1.1, 0.95, 0.11),
        TokenUncertaintyStat("approved", -0.2, 0.35, 0.62),
        TokenUncertaintyStat("the", -0.1, 0.2, 0.7),
        TokenUncertaintyStat("proposal", -0.25, 0.3, 0.58),
    ]
    internal_signal = InternalModelSignal(
        last_layer_pooled_l2=1.0,
        last_layer_pooled_mean_abs=0.35,
        selected_layer_norm_variance=0.06,
        layer_disagreement_mean=0.14,
        selected_layer_disagreement_max=0.19,
        early_late_layer_consistency=0.79,
    )

    features = extract_specialist_features(
        prompt="How many delegates did Paris and Berlin report in 2024?",
        response="In 2024 Paris reported 43 delegates and Berlin approved the proposal.",
        token_stats=token_stats,
        internal_signal=internal_signal,
    )

    assert "specialist_numeric_small_delta_proxy" in features
    assert "specialist_numeric_local_margin_dip" in features
    assert "specialist_entity_margin_variance" in features
    assert "specialist_entity_local_instability" in features
    assert "specialist_long_sliding_entropy_max" in features
    assert "specialist_long_local_anomaly_peak" in features
    assert "specialist_local_uncertainty_spike" in features
    assert "specialist_local_margin_dip_contrast" in features
    assert all(value == value for value in features.values())


def test_extract_specialist_features_increase_local_anomaly_scores_for_spiky_sequences() -> None:
    stable_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.25, 0.65),
        TokenUncertaintyStat("has", -0.2, 0.22, 0.63),
        TokenUncertaintyStat("42", -0.25, 0.24, 0.61),
        TokenUncertaintyStat("delegates", -0.2, 0.23, 0.64),
        TokenUncertaintyStat("today", -0.2, 0.21, 0.66),
    ]
    spiky_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.25, 0.65),
        TokenUncertaintyStat("has", -0.2, 0.22, 0.63),
        TokenUncertaintyStat("42", -1.4, 1.1, 0.08),
        TokenUncertaintyStat("delegates", -1.0, 0.95, 0.1),
        TokenUncertaintyStat("today", -0.2, 0.21, 0.66),
    ]

    stable = extract_specialist_features(
        prompt="How many delegates does Paris have today?",
        response="Paris has 42 delegates today.",
        token_stats=stable_stats,
        internal_signal=None,
    )
    spiky = extract_specialist_features(
        prompt="How many delegates does Paris have today?",
        response="Paris has 42 delegates today.",
        token_stats=spiky_stats,
        internal_signal=None,
    )

    assert spiky["specialist_numeric_local_margin_dip"] > stable["specialist_numeric_local_margin_dip"]
    assert spiky["specialist_local_uncertainty_spike"] > stable["specialist_local_uncertainty_spike"]
    assert spiky["specialist_long_local_anomaly_peak"] > stable["specialist_long_local_anomaly_peak"]
