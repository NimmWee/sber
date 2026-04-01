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
