from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from models.head import TrainedLightGBMHead, train_lightgbm_head


def test_train_lightgbm_head_predicts_and_roundtrips(tmp_path) -> None:
    feature_rows = [
        {"token_mean_logprob": -0.1, "token_entropy_mean": 0.2, "internal_layer_disagreement_mean": 0.05},
        {"token_mean_logprob": -1.1, "token_entropy_mean": 0.9, "internal_layer_disagreement_mean": 0.4},
        {"token_mean_logprob": -0.2, "token_entropy_mean": 0.25, "internal_layer_disagreement_mean": 0.08},
        {"token_mean_logprob": -1.3, "token_entropy_mean": 1.0, "internal_layer_disagreement_mean": 0.5},
    ]
    labels = [0, 1, 0, 1]

    head = train_lightgbm_head(feature_rows, labels)

    probabilities = head.predict_proba_batch(feature_rows)

    assert len(probabilities) == 4
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    artifact_path = tmp_path / "lightgbm_head.json"
    head.save(artifact_path)
    loaded_head = TrainedLightGBMHead.load(artifact_path)

    reloaded_probabilities = loaded_head.predict_proba_batch(feature_rows)
    assert reloaded_probabilities == probabilities


def test_train_lightgbm_head_accepts_sample_weights() -> None:
    feature_rows = [
        {"token_mean_logprob": -0.2, "token_entropy_mean": 0.25},
        {"token_mean_logprob": -0.25, "token_entropy_mean": 0.3},
        {"token_mean_logprob": -1.0, "token_entropy_mean": 0.8},
        {"token_mean_logprob": -1.1, "token_entropy_mean": 0.9},
    ]
    labels = [0, 0, 1, 1]
    sample_weights = [0.5, 0.5, 1.2, 1.2]

    head = train_lightgbm_head(
        feature_rows,
        labels,
        sample_weights=sample_weights,
    )

    probabilities = head.predict_proba_batch(feature_rows)
    assert len(probabilities) == 4
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
