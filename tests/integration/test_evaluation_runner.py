import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.runner import (
    EvaluationDataset,
    EvaluationRunner,
    EvaluationSplit,
    RawExampleEvaluationDataset,
    RawLabeledExample,
    TrainValidationEvaluationRunner,
)
from features.extractor import StructuralFeatureExtractor
from features.extractor import TokenUncertaintyStat
from inference.token_stats import TransformersProviderConfig, TransformersTokenStatProvider


class FakeTokenizer:
    def __init__(self) -> None:
        self.bos_token_id = 0
        self._token_to_id = {
            "<bos>": 0,
            "hello": 1,
            "prompt": 2,
            "world": 3,
            "again": 4,
            "2024": 5,
        }
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}

    def __call__(self, text: str, add_special_tokens: bool = False):
        tokens = text.split() if text else []
        return {"input_ids": [self._token_to_id[token] for token in tokens]}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]


class FakeModelOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class FakeModel:
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> FakeModelOutput:
        sequence = input_ids[0].tolist()
        vocab_size = 6
        logits = torch.zeros((1, len(sequence), vocab_size), dtype=torch.float32)
        for index in range(len(sequence) - 1):
            next_token_id = sequence[index + 1]
            logits[0, index, next_token_id] = 5.0
            logits[0, index, (next_token_id + 1) % vocab_size] = 4.0
        hidden_states = None
        if output_hidden_states:
            hidden_states = (
                torch.zeros((1, len(sequence), 3), dtype=torch.float32),
                torch.tensor(
                    [
                        [
                            [0.10, 0.00, 0.00],
                            [0.20, 0.00, 0.00],
                            [0.50, 0.20, 0.00],
                            [0.60, 0.10, 0.00],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        [
                            [0.00, 0.10, 0.00],
                            [0.00, 0.20, 0.00],
                            [0.40, 0.30, 0.10],
                            [0.55, 0.12, 0.08],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        [
                            [0.00, 0.00, 0.10],
                            [0.00, 0.00, 0.20],
                            [0.60, 0.10, 0.05],
                            [0.70, 0.08, 0.05],
                        ]
                    ],
                    dtype=torch.float32,
                ),
            )
        return FakeModelOutput(logits, hidden_states=hidden_states)


class InMemoryEvaluationDataset(EvaluationDataset):
    def load_split(self) -> EvaluationSplit:
        return EvaluationSplit(
            train_features=[
                {"response_length": 0.1, "digit_count": 0.0, "novelty_ratio_proxy": 0.1},
                {"response_length": 0.2, "digit_count": 0.0, "novelty_ratio_proxy": 0.2},
                {"response_length": 0.8, "digit_count": 1.0, "novelty_ratio_proxy": 0.8},
                {"response_length": 0.9, "digit_count": 1.0, "novelty_ratio_proxy": 0.9},
            ],
            train_labels=[0, 0, 1, 1],
            validation_features=[
                {"response_length": 0.15, "digit_count": 0.0, "novelty_ratio_proxy": 0.15},
                {"response_length": 0.85, "digit_count": 1.0, "novelty_ratio_proxy": 0.85},
            ],
            validation_labels=[0, 1],
        )


def test_evaluation_runner_trains_evaluates_and_saves_summary(tmp_path) -> None:
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=InMemoryEvaluationDataset(),
        artifact_dir=tmp_path,
    )

    summary = runner.run()

    assert 0.0 <= summary.pr_auc <= 1.0
    assert summary.sample_size == 2
    assert summary.model_artifact_path is not None
    assert summary.summary_artifact_path is not None
    assert Path(summary.model_artifact_path).exists()
    assert Path(summary.summary_artifact_path).exists()

    summary_payload = json.loads(Path(summary.summary_artifact_path).read_text())
    assert summary_payload["pr_auc"] == summary.pr_auc


def test_evaluation_runner_from_raw_examples_builds_features_and_saves_artifacts(
    tmp_path,
) -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="Who wrote Hamlet?",
                response="William Shakespeare wrote Hamlet.",
                label=0,
            ),
            RawLabeledExample(
                prompt="What year was the treaty signed?",
                response="The treaty was signed in 1492.",
                label=1,
            ),
            RawLabeledExample(
                prompt="Name the capital of France.",
                response="Paris is the capital of France.",
                label=0,
            ),
            RawLabeledExample(
                prompt="How many moons does Mars have?",
                response="Mars has 12 moons.",
                label=1,
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="What is the launch date?",
                response="The launch date was 2024-01-15.",
                label=1,
            ),
            RawLabeledExample(
                prompt="What is the capital of Italy?",
                response="Rome is the capital of Italy.",
                label=0,
            ),
        ],
        extractor=StructuralFeatureExtractor(),
    )
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    )

    summary = runner.run()

    assert 0.0 <= summary.pr_auc <= 1.0
    assert summary.sample_size == 2
    assert summary.model_artifact_path is not None
    assert summary.summary_artifact_path is not None
    assert Path(summary.model_artifact_path).exists()
    assert Path(summary.summary_artifact_path).exists()


def test_evaluation_runner_with_uncertainty_proxies_uses_extended_schema(
    tmp_path,
) -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="Who wrote Hamlet?",
                response="William Shakespeare wrote Hamlet in 1603.",
                label=0,
            ),
            RawLabeledExample(
                prompt="What year was the treaty signed?",
                response="The treaty was signed in 1492 and amended on 2024-01-15.",
                label=1,
            ),
            RawLabeledExample(
                prompt="Name the capital of France.",
                response="Paris is the capital of France.",
                label=0,
            ),
            RawLabeledExample(
                prompt="How many moons does Mars have?",
                response="Mars has 12 moons according to this answer.",
                label=1,
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="What is the launch date?",
                response="The launch date was 2024-01-15.",
                label=1,
            ),
            RawLabeledExample(
                prompt="What is the capital of Italy?",
                response="Rome is the capital of Italy.",
                label=0,
            ),
        ],
        extractor=StructuralFeatureExtractor(enable_uncertainty_proxies=True),
    )
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    )

    summary = runner.run()
    model_payload = json.loads(Path(summary.model_artifact_path).read_text())

    assert 0.0 <= summary.pr_auc <= 1.0
    assert "numeric_density" in model_payload["feature_names"]
    assert "entity_like_token_density_proxy" in model_payload["feature_names"]
    assert "prompt_response_lexical_divergence" in model_payload["feature_names"]


def test_evaluation_runner_with_token_uncertainty_uses_extended_schema(
    tmp_path,
) -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="Who wrote Hamlet?",
                response="William Shakespeare wrote Hamlet in 1603.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("William", -0.05, 0.12, 0.75),
                    TokenUncertaintyStat("Shakespeare", -0.12, 0.18, 0.63),
                    TokenUncertaintyStat("1603", -0.85, 0.55, 0.20),
                ],
            ),
            RawLabeledExample(
                prompt="What year was the treaty signed?",
                response="The treaty was signed in 1492 and amended on 2024-01-15.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("1492", -1.10, 0.82, 0.10),
                    TokenUncertaintyStat("2024-01-15", -1.25, 0.95, 0.08),
                    TokenUncertaintyStat("signed", -0.20, 0.22, 0.52),
                ],
            ),
            RawLabeledExample(
                prompt="Name the capital of France.",
                response="Paris is the capital of France.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Paris", -0.04, 0.11, 0.76),
                    TokenUncertaintyStat("capital", -0.08, 0.15, 0.69),
                    TokenUncertaintyStat("France", -0.06, 0.13, 0.72),
                ],
            ),
            RawLabeledExample(
                prompt="How many moons does Mars have?",
                response="Mars has 12 moons according to this answer.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("Mars", -0.06, 0.13, 0.73),
                    TokenUncertaintyStat("12", -1.00, 0.88, 0.09),
                    TokenUncertaintyStat("answer", -0.50, 0.44, 0.22),
                ],
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="What is the launch date?",
                response="The launch date was 2024-01-15.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("launch", -0.22, 0.24, 0.48),
                    TokenUncertaintyStat("2024-01-15", -1.15, 0.90, 0.08),
                ],
            ),
            RawLabeledExample(
                prompt="What is the capital of Italy?",
                response="Rome is the capital of Italy.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Rome", -0.04, 0.10, 0.78),
                    TokenUncertaintyStat("capital", -0.07, 0.14, 0.70),
                ],
            ),
        ],
        extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
    )
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    )

    summary = runner.run()
    model_payload = json.loads(Path(summary.model_artifact_path).read_text())

    assert 0.0 <= summary.pr_auc <= 1.0
    assert "token_mean_logprob" in model_payload["feature_names"]
    assert "token_min_logprob" in model_payload["feature_names"]
    assert "token_entropy_mean" in model_payload["feature_names"]
    assert "token_logprob_std" in model_payload["feature_names"]
    assert "token_tail_low_confidence_rate_le_1_0" in model_payload["feature_names"]
    assert "token_entity_like_entropy_mean" in model_payload["feature_names"]


def test_evaluation_runner_with_provider_backed_token_stats_uses_extended_schema(
    tmp_path,
) -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(model_id="distilgpt2"),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
        ],
        extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
        token_stat_provider=provider,
    )
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    )

    summary = runner.run()
    model_payload = json.loads(Path(summary.model_artifact_path).read_text())

    assert 0.0 <= summary.pr_auc <= 1.0
    assert "token_mean_logprob" in model_payload["feature_names"]
    assert "token_min_logprob" in model_payload["feature_names"]
    assert "token_entropy_mean" in model_payload["feature_names"]
    assert "token_logprob_std" in model_payload["feature_names"]
    assert "token_tail_low_confidence_rate_le_1_0" in model_payload["feature_names"]
    assert "token_entity_like_entropy_mean" in model_payload["feature_names"]


def test_evaluation_runner_with_provider_backed_internal_features_uses_extended_schema(
    tmp_path,
) -> None:
    provider = TransformersTokenStatProvider(
        config=TransformersProviderConfig(
            model_id="distilgpt2",
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="hello prompt",
                response="world again",
                label=0,
            ),
            RawLabeledExample(
                prompt="hello prompt",
                response="world 2024",
                label=1,
            ),
        ],
        extractor=StructuralFeatureExtractor(
            enable_token_uncertainty=True,
            enable_internal_features=True,
        ),
        token_stat_provider=provider,
    )
    runner: EvaluationRunner = TrainValidationEvaluationRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    )

    summary = runner.run()
    model_payload = json.loads(Path(summary.model_artifact_path).read_text())

    assert 0.0 <= summary.pr_auc <= 1.0
    assert "internal_last_layer_pooled_l2" in model_payload["feature_names"]
    assert "internal_selected_layer_norm_variance" in model_payload["feature_names"]
    assert "internal_layer_disagreement_mean" in model_payload["feature_names"]
