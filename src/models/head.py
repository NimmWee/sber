import json
import math
from pathlib import Path
from typing import Mapping, Protocol


class ScoringHead(Protocol):
    def predict_proba(self, features: Mapping[str, float]) -> float:
        ...


class LinearScoringHead:
    _WEIGHTS = {
        "response_length": 0.002,
        "token_count_proxy": 0.015,
        "digit_count": 0.04,
        "punctuation_count": 0.03,
        "sentence_count_proxy": 0.05,
        "prompt_response_overlap": -1.0,
        "novelty_ratio_proxy": 1.0,
    }
    _BIAS = -1.5

    def predict_proba(self, features: Mapping[str, float]) -> float:
        linear_score = self._BIAS
        for feature_name, weight in self._WEIGHTS.items():
            linear_score += float(features.get(feature_name, 0.0)) * weight

        bounded_score = max(min(linear_score, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-bounded_score))


class TrainedLogisticRegressionHead:
    def __init__(
        self,
        feature_names: tuple[str, ...],
        weights: tuple[float, ...],
        bias: float,
    ) -> None:
        self.feature_names = feature_names
        self.weights = weights
        self.bias = bias

    def predict_proba(self, features: Mapping[str, float]) -> float:
        linear_score = self.bias
        for feature_name, weight in zip(self.feature_names, self.weights):
            linear_score += float(features.get(feature_name, 0.0)) * weight

        bounded_score = max(min(linear_score, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-bounded_score))

    def predict_proba_batch(
        self, feature_rows: list[Mapping[str, float]]
    ) -> list[float]:
        return [self.predict_proba(feature_row) for feature_row in feature_rows]

    def save(self, path: str | Path) -> None:
        artifact_path = Path(path)
        artifact_path.write_text(
            json.dumps(
                {
                    "feature_names": list(self.feature_names),
                    "weights": list(self.weights),
                    "bias": self.bias,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedLogisticRegressionHead":
        artifact = json.loads(Path(path).read_text())
        return cls(
            feature_names=tuple(artifact["feature_names"]),
            weights=tuple(float(weight) for weight in artifact["weights"]),
            bias=float(artifact["bias"]),
        )


def train_logistic_regression_head(
    feature_rows: list[Mapping[str, float]],
    labels: list[int],
    *,
    epochs: int = 250,
    learning_rate: float = 0.1,
) -> TrainedLogisticRegressionHead:
    if len(feature_rows) != len(labels):
        raise ValueError("feature_rows and labels must have the same length")
    if not feature_rows:
        raise ValueError("feature_rows must not be empty")

    feature_names = tuple(
        sorted(
            {
                feature_name
                for feature_row in feature_rows
                for feature_name in feature_row
            }
        )
    )
    weights = [0.0 for _ in feature_names]
    bias = 0.0

    for _ in range(epochs):
        weight_gradients = [0.0 for _ in feature_names]
        bias_gradient = 0.0

        for feature_row, label in zip(feature_rows, labels):
            probability = _predict_probability(
                feature_names=feature_names,
                weights=weights,
                bias=bias,
                feature_row=feature_row,
            )
            error = probability - float(label)

            for index, feature_name in enumerate(feature_names):
                weight_gradients[index] += error * float(
                    feature_row.get(feature_name, 0.0)
                )
            bias_gradient += error

        scale = 1.0 / len(feature_rows)
        for index in range(len(weights)):
            weights[index] -= learning_rate * weight_gradients[index] * scale
        bias -= learning_rate * bias_gradient * scale

    return TrainedLogisticRegressionHead(
        feature_names=feature_names,
        weights=tuple(weights),
        bias=bias,
    )


def _predict_probability(
    *,
    feature_names: tuple[str, ...],
    weights: list[float],
    bias: float,
    feature_row: Mapping[str, float],
) -> float:
    linear_score = bias
    for feature_name, weight in zip(feature_names, weights):
        linear_score += float(feature_row.get(feature_name, 0.0)) * weight

    bounded_score = max(min(linear_score, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-bounded_score))
