from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.specialist_ensemble import (
    build_fusion_feature_row,
    extract_specialist_features,
    select_important_specialist_features,
)
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


def test_select_important_specialist_features_keeps_required_and_top_ranked_features() -> None:
    selected = select_important_specialist_features(
        all_feature_names=[
            "specialist_numeric_density",
            "specialist_numeric_margin_min",
            "specialist_numeric_local_entropy_spike",
            "specialist_local_uncertainty_spike",
            "specialist_entity_local_instability",
        ],
        feature_importance=[
            {"feature_name": "specialist_numeric_margin_min", "importance": 9.0},
            {"feature_name": "specialist_entity_local_instability", "importance": 7.0},
            {"feature_name": "specialist_numeric_local_entropy_spike", "importance": 0.2},
            {"feature_name": "specialist_local_uncertainty_spike", "importance": 0.1},
        ],
        required_feature_names=["specialist_numeric_density"],
        max_selected_features=3,
    )

    assert "specialist_numeric_density" in selected
    assert "specialist_numeric_margin_min" in selected
    assert "specialist_entity_local_instability" in selected
    assert "specialist_numeric_local_entropy_spike" not in selected
    assert "specialist_local_uncertainty_spike" not in selected


def test_extract_specialist_features_adds_consistency_signals_for_segment_drift() -> None:
    coherent_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.2, 0.7),
        TokenUncertaintyStat("is", -0.15, 0.18, 0.68),
        TokenUncertaintyStat("the", -0.12, 0.16, 0.69),
        TokenUncertaintyStat("capital", -0.16, 0.2, 0.66),
        TokenUncertaintyStat("of", -0.12, 0.16, 0.69),
        TokenUncertaintyStat("France", -0.22, 0.22, 0.64),
    ]
    drifting_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.2, 0.7),
        TokenUncertaintyStat("is", -0.15, 0.18, 0.68),
        TokenUncertaintyStat("the", -0.12, 0.16, 0.69),
        TokenUncertaintyStat("capital", -1.0, 0.95, 0.12),
        TokenUncertaintyStat("of", -0.12, 0.16, 0.69),
        TokenUncertaintyStat("Germany", -1.2, 1.05, 0.08),
    ]

    coherent = extract_specialist_features(
        prompt="What is the capital of France?",
        response="Paris is the capital of France.",
        token_stats=coherent_stats,
        internal_signal=None,
    )
    drifting = extract_specialist_features(
        prompt="What is the capital of France?",
        response="Paris is the capital of Germany.",
        token_stats=drifting_stats,
        internal_signal=None,
    )

    assert "specialist_consistency_segment_disagreement" in coherent
    assert "specialist_consistency_prompt_alignment" in coherent
    assert "specialist_consistency_prompt_drift" in coherent
    assert drifting["specialist_consistency_segment_disagreement"] > coherent["specialist_consistency_segment_disagreement"]
    assert drifting["specialist_consistency_prompt_drift"] > coherent["specialist_consistency_prompt_drift"]


def test_extract_specialist_features_adds_prompt_stability_signals_from_perturbed_pass() -> None:
    original_stats = [
        TokenUncertaintyStat("Paris", -0.2, 0.2, 0.7),
        TokenUncertaintyStat("France", -0.22, 0.22, 0.64),
    ]
    perturbed_stats = [
        TokenUncertaintyStat("Paris", -0.9, 0.85, 0.12),
        TokenUncertaintyStat("France", -0.95, 0.88, 0.1),
    ]

    stable = extract_specialist_features(
        prompt="What is the capital of France?",
        response="Paris, France.",
        token_stats=original_stats,
        internal_signal=None,
        perturbed_token_stats=original_stats,
        perturbed_internal_signal=None,
    )
    unstable = extract_specialist_features(
        prompt="What is the capital of France?",
        response="Paris, France.",
        token_stats=original_stats,
        internal_signal=None,
        perturbed_token_stats=perturbed_stats,
        perturbed_internal_signal=None,
    )

    assert "specialist_stability_logprob_shift" in stable
    assert "specialist_stability_entropy_shift" in stable
    assert "specialist_stability_margin_shift" in stable
    assert unstable["specialist_stability_logprob_shift"] > stable["specialist_stability_logprob_shift"]
    assert unstable["specialist_stability_entropy_shift"] > stable["specialist_stability_entropy_shift"]
