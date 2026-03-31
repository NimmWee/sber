import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from eval.metrics import compute_pr_auc
from features.extractor import (
    FeatureExtractor,
    InternalModelSignal,
    TokenUncertaintyStat,
)
from inference.token_stats import TokenStatProvider
from models.head import train_logistic_regression_head


@dataclass(frozen=True)
class EvalSummary:
    pr_auc: float
    sample_size: int
    model_artifact_path: str | None = None
    summary_artifact_path: str | None = None


class EvaluationRunner(Protocol):
    def run(self) -> EvalSummary:
        ...


@dataclass(frozen=True)
class EvaluationSplit:
    train_features: list[Mapping[str, float]]
    train_labels: list[int]
    validation_features: list[Mapping[str, float]]
    validation_labels: list[int]


class EvaluationDataset(Protocol):
    def load_split(self) -> EvaluationSplit:
        ...


@dataclass(frozen=True)
class RawLabeledExample:
    prompt: str
    response: str
    label: int
    token_stats: list[TokenUncertaintyStat] | None = None
    internal_signal: InternalModelSignal | None = None


class RawExampleEvaluationDataset:
    def __init__(
        self,
        *,
        train_examples: list[RawLabeledExample],
        validation_examples: list[RawLabeledExample],
        extractor: FeatureExtractor,
        token_stat_provider: TokenStatProvider | None = None,
    ) -> None:
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.extractor = extractor
        self.token_stat_provider = token_stat_provider

    def load_split(self) -> EvaluationSplit:
        if not self.train_examples and not self.validation_examples:
            raise ValueError("raw example datasets must not be empty")

        train_features, train_labels = self._prepare_examples(self.train_examples)
        validation_features, validation_labels = self._prepare_examples(
            self.validation_examples
        )
        return EvaluationSplit(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )

    def _prepare_examples(
        self, examples: list[RawLabeledExample]
    ) -> tuple[list[Mapping[str, float]], list[int]]:
        prepared_features: list[Mapping[str, float]] = []
        prepared_labels: list[int] = []

        for example in examples:
            if not isinstance(example, RawLabeledExample):
                raise TypeError("examples must contain RawLabeledExample instances")
            if example.label not in (0, 1):
                raise ValueError("label must be 0 or 1")
            token_stats = example.token_stats
            internal_signal = example.internal_signal
            if (
                (token_stats is None or internal_signal is None)
                and self.token_stat_provider is not None
            ):
                collected = self.token_stat_provider.collect_signals(
                    prompt=example.prompt,
                    response=example.response,
                )
                if token_stats is None:
                    token_stats = collected.token_stats
                if internal_signal is None:
                    internal_signal = collected.internal_signal

            prepared_features.append(
                self.extractor.extract(
                    prompt=example.prompt,
                    response=example.response,
                    token_stats=token_stats,
                    internal_signal=internal_signal,
                )
            )
            prepared_labels.append(example.label)

        return prepared_features, prepared_labels


class TrainValidationEvaluationRunner:
    def __init__(self, dataset: EvaluationDataset, artifact_dir: str | Path) -> None:
        self.dataset = dataset
        self.artifact_dir = Path(artifact_dir)

    def run(self) -> EvalSummary:
        split = self.dataset.load_split()
        model = train_logistic_regression_head(split.train_features, split.train_labels)
        probabilities = model.predict_proba_batch(split.validation_features)
        pr_auc = compute_pr_auc(split.validation_labels, probabilities)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        model_artifact_path = self.artifact_dir / "logistic_head.json"
        summary_artifact_path = self.artifact_dir / "eval_summary.json"

        model.save(model_artifact_path)
        summary = EvalSummary(
            pr_auc=pr_auc,
            sample_size=len(split.validation_labels),
            model_artifact_path=str(model_artifact_path),
            summary_artifact_path=str(summary_artifact_path),
        )
        summary_artifact_path.write_text(json.dumps(asdict(summary), indent=2))
        return summary
