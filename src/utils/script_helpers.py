import json
from pathlib import Path

from eval.runner import RawLabeledExample
from inference.token_stats import TransformersProviderConfig


def load_transformers_provider_config(
    path: str | Path,
) -> TransformersProviderConfig:
    return TransformersProviderConfig.from_json(path)


def write_json_artifact(
    *,
    artifact_dir: str | Path,
    filename: str,
    payload: dict,
) -> Path:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / filename
    artifact_path.write_text(json.dumps(payload, indent=2))
    return artifact_path


def build_smoke_examples() -> tuple[list[RawLabeledExample], list[RawLabeledExample]]:
    train_examples = [
        RawLabeledExample(
            prompt="Who wrote Hamlet?",
            response="William Shakespeare wrote Hamlet.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of France?",
            response="Paris is the capital of France.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of Italy?",
            response="Rome is the capital of Italy.",
            label=0,
        ),
        RawLabeledExample(
            prompt="How many moons does Mars have?",
            response="Mars has 2 moons.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of Italy?",
            response="The capital of Italy is Verona.",
            label=1,
        ),
        RawLabeledExample(
            prompt="How many moons does Mars have?",
            response="Mars has 12 moons and 2 invisible rings.",
            label=1,
        ),
        RawLabeledExample(
            prompt="When was the treaty signed?",
            response="The treaty was signed on 1492-13-40.",
            label=1,
        ),
        RawLabeledExample(
            prompt="Who wrote Hamlet?",
            response="Hamlet was written by Napoleon Bonaparte in 2024.",
            label=1,
        ),
    ]
    validation_examples = [
        RawLabeledExample(
            prompt="What is the capital of Germany?",
            response="Berlin is the capital of Germany.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the boiling point of water at sea level?",
            response="Water boils at 100 degrees Celsius at sea level.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of Germany?",
            response="The capital of Germany is Munich.",
            label=1,
        ),
        RawLabeledExample(
            prompt="What is the boiling point of water at sea level?",
            response="Water boils at 250 degrees Celsius at sea level.",
            label=1,
        ),
    ]
    return train_examples, validation_examples
