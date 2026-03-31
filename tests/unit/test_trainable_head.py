import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from models.head import TrainedLogisticRegressionHead, train_logistic_regression_head


def _training_rows() -> tuple[list[dict[str, float]], list[int]]:
    feature_rows = [
        {"response_length": 0.1, "digit_count": 0.0, "novelty_ratio_proxy": 0.1},
        {"response_length": 0.2, "digit_count": 0.0, "novelty_ratio_proxy": 0.2},
        {"response_length": 0.8, "digit_count": 1.0, "novelty_ratio_proxy": 0.8},
        {"response_length": 0.9, "digit_count": 1.0, "novelty_ratio_proxy": 0.9},
    ]
    labels = [0, 0, 1, 1]
    return feature_rows, labels


def test_training_pipeline_accepts_feature_rows_and_labels() -> None:
    feature_rows, labels = _training_rows()

    head = train_logistic_regression_head(feature_rows, labels)

    assert isinstance(head, TrainedLogisticRegressionHead)


def test_trained_head_returns_bounded_probabilities() -> None:
    feature_rows, labels = _training_rows()
    head = train_logistic_regression_head(feature_rows, labels)

    probabilities = head.predict_proba_batch(feature_rows)

    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_trained_head_prediction_shape_matches_input_rows() -> None:
    feature_rows, labels = _training_rows()
    head = train_logistic_regression_head(feature_rows, labels)

    probabilities = head.predict_proba_batch(feature_rows)

    assert len(probabilities) == len(feature_rows)


def test_model_artifact_save_load_roundtrip(tmp_path) -> None:
    feature_rows, labels = _training_rows()
    head = train_logistic_regression_head(feature_rows, labels)
    artifact_path = tmp_path / "logistic_head.json"

    head.save(artifact_path)
    loaded = TrainedLogisticRegressionHead.load(artifact_path)

    original_probabilities = head.predict_proba_batch(feature_rows)
    loaded_probabilities = loaded.predict_proba_batch(feature_rows)

    assert artifact_path.exists()
    assert all(
        math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9)
        for first, second in zip(original_probabilities, loaded_probabilities)
    )


def test_loaded_model_keeps_feature_schema_for_inference_with_reordered_keys(
    tmp_path,
) -> None:
    feature_rows, labels = _training_rows()
    head = train_logistic_regression_head(feature_rows, labels)
    artifact_path = tmp_path / "logistic_head.json"

    head.save(artifact_path)
    loaded = TrainedLogisticRegressionHead.load(artifact_path)
    reordered_features = {
        "novelty_ratio_proxy": 0.8,
        "digit_count": 1.0,
        "response_length": 0.8,
    }

    original_probability = head.predict_proba(reordered_features)
    loaded_probability = loaded.predict_proba(reordered_features)

    assert loaded.feature_names == head.feature_names
    assert math.isclose(
        original_probability,
        loaded_probability,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
