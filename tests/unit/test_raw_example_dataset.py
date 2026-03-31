from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.runner import RawExampleEvaluationDataset, RawLabeledExample
from features.extractor import StructuralFeatureExtractor


def test_raw_example_dataset_accepts_prompt_response_and_label() -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(prompt="Question?", response="Answer.", label=0),
        ],
        validation_examples=[
            RawLabeledExample(prompt="Question 2?", response="Answer 2.", label=1),
        ],
        extractor=StructuralFeatureExtractor(),
    )

    split = dataset.load_split()

    assert split.train_labels == [0]
    assert split.validation_labels == [1]


def test_feature_extraction_is_applied_consistently_to_train_and_validation() -> None:
    extractor = StructuralFeatureExtractor()
    train_example = RawLabeledExample(
        prompt="What year was it signed?",
        response="It was signed in 2024.",
        label=0,
    )
    validation_example = RawLabeledExample(
        prompt="How many samples returned?",
        response="The mission returned 3 samples.",
        label=1,
    )
    dataset = RawExampleEvaluationDataset(
        train_examples=[train_example],
        validation_examples=[validation_example],
        extractor=extractor,
    )

    split = dataset.load_split()

    assert dict(split.train_features[0]) == dict(
        extractor.extract(train_example.prompt, train_example.response)
    )
    assert dict(split.validation_features[0]) == dict(
        extractor.extract(validation_example.prompt, validation_example.response)
    )


def test_label_and_row_counts_match_after_preparation() -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(prompt="P1", response="R1", label=0),
            RawLabeledExample(prompt="P2", response="R2", label=1),
        ],
        validation_examples=[
            RawLabeledExample(prompt="P3", response="R3", label=0),
        ],
        extractor=StructuralFeatureExtractor(),
    )

    split = dataset.load_split()

    assert len(split.train_features) == len(split.train_labels) == 2
    assert len(split.validation_features) == len(split.validation_labels) == 1


def test_invalid_labels_are_rejected() -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(prompt="Prompt", response="Response", label=2),
        ],
        validation_examples=[
            RawLabeledExample(prompt="Prompt", response="Response", label=1),
        ],
        extractor=StructuralFeatureExtractor(),
    )

    with pytest.raises(ValueError, match="label"):
        dataset.load_split()


def test_empty_dataset_fails_clearly() -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[],
        validation_examples=[],
        extractor=StructuralFeatureExtractor(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        dataset.load_split()


def test_malformed_examples_fail_clearly() -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[{"prompt": "Prompt only"}],  # type: ignore[list-item]
        validation_examples=[
            RawLabeledExample(prompt="Prompt", response="Response", label=1),
        ],
        extractor=StructuralFeatureExtractor(),
    )

    with pytest.raises(TypeError, match="RawLabeledExample"):
        dataset.load_split()
