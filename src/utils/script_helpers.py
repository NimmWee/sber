import json
from pathlib import Path

from eval.runner import RawLabeledExample
from inference.token_stats import TransformersProviderConfig


def load_transformers_provider_config(
    path: str | Path,
) -> TransformersProviderConfig:
    return TransformersProviderConfig.from_json(path)


def resolve_transformers_provider_config(
    *,
    project_root: str | Path,
    explicit_config_path: str | Path | None = None,
) -> TransformersProviderConfig:
    if explicit_config_path is not None:
        return load_transformers_provider_config(explicit_config_path)

    project_root_path = Path(project_root)
    local_config_path = project_root_path / "configs" / "token_stat_provider.local.json"
    default_config_path = project_root_path / "configs" / "token_stat_provider.json"

    if local_config_path.exists():
        local_config = load_transformers_provider_config(local_config_path)
        if local_config.checkpoint_path is not None and Path(
            local_config.checkpoint_path
        ).exists():
            return local_config

    return load_transformers_provider_config(default_config_path)


def resolve_public_benchmark_path(
    *,
    project_root: str | Path,
    explicit_dataset_path: str | Path | None = None,
) -> Path:
    if explicit_dataset_path is not None:
        return Path(explicit_dataset_path)

    project_root_path = Path(project_root)
    root_candidate = project_root_path / "knowledge_bench_public.csv"
    if root_candidate.exists():
        return root_candidate

    data_candidate = project_root_path / "data" / "knowledge_bench_public.csv"
    if data_candidate.exists():
        return data_candidate

    raise FileNotFoundError("knowledge_bench_public.csv was not found")


def resolve_text_training_seed_path(
    *,
    project_root: str | Path,
    explicit_seed_path: str | Path | None = None,
) -> Path:
    if explicit_seed_path is not None:
        return Path(explicit_seed_path)

    project_root_path = Path(project_root)
    default_candidate = project_root_path / "data" / "public_seed_facts.jsonl"
    if default_candidate.exists():
        return default_candidate

    raise FileNotFoundError("public_seed_facts.jsonl was not found")


def resolve_triviaqa_path(
    *,
    project_root: str | Path,
    explicit_dataset_path: str | Path | None = None,
) -> Path:
    if explicit_dataset_path is not None:
        return Path(explicit_dataset_path)

    project_root_path = Path(project_root)
    candidates = [
        project_root_path / "data" / "triviaqa.jsonl",
        project_root_path / "data" / "triviaqa.json",
        project_root_path / "data" / "textual" / "triviaqa.jsonl",
        project_root_path / "data" / "triviaqa" / "triviaqa.jsonl",
        project_root_path / "data" / "triviaqa" / "triviaqa.json",
        project_root_path / "data" / "triviaqa" / "unfiltered-web-dev.json",
        project_root_path / "data" / "triviaqa" / "wikipedia-dev.json",
        project_root_path / "trivia-qa" / "pair" / "train-00000-of-00001.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    data_root = project_root_path / "data"
    if data_root.exists():
        for candidate in sorted(data_root.rglob("*triviaqa*.json*")):
            if candidate.is_file():
                return candidate

    triviaqa_root = project_root_path / "trivia-qa"
    if triviaqa_root.exists():
        for candidate in sorted(triviaqa_root.rglob("*.parquet")):
            if candidate.is_file():
                return candidate

    raise FileNotFoundError("local TriviaQA dataset was not found")


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


def build_ablation_examples() -> tuple[list[RawLabeledExample], list[RawLabeledExample]]:
    train_examples, validation_examples = build_smoke_examples()
    train_examples = train_examples + [
        RawLabeledExample(
            prompt="Who painted the Mona Lisa?",
            response="Leonardo da Vinci painted the Mona Lisa.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of Japan?",
            response="Tokyo is the capital of Japan.",
            label=0,
        ),
        RawLabeledExample(
            prompt="When did Apollo 11 land on the Moon?",
            response="Apollo 11 landed on the Moon in 1969.",
            label=0,
        ),
        RawLabeledExample(
            prompt="How many planets are in the Solar System?",
            response="There are 8 planets in the Solar System.",
            label=0,
        ),
        RawLabeledExample(
            prompt="Who painted the Mona Lisa?",
            response="Pablo Picasso painted the Mona Lisa in 1921.",
            label=1,
        ),
        RawLabeledExample(
            prompt="What is the capital of Japan?",
            response="Osaka is the capital of Japan as of 2026.",
            label=1,
        ),
        RawLabeledExample(
            prompt="When did Apollo 11 land on the Moon?",
            response="Apollo 11 landed on the Moon in 1974.",
            label=1,
        ),
        RawLabeledExample(
            prompt="How many planets are in the Solar System?",
            response="There are 11 planets in the Solar System today.",
            label=1,
        ),
    ]
    validation_examples = validation_examples + [
        RawLabeledExample(
            prompt="Who discovered penicillin?",
            response="Alexander Fleming discovered penicillin.",
            label=0,
        ),
        RawLabeledExample(
            prompt="What is the capital of Canada?",
            response="Ottawa is the capital of Canada.",
            label=0,
        ),
        RawLabeledExample(
            prompt="Who discovered penicillin?",
            response="Marie Curie discovered penicillin in 1932.",
            label=1,
        ),
        RawLabeledExample(
            prompt="What is the capital of Canada?",
            response="Toronto is the capital of Canada.",
            label=1,
        ),
    ]
    return train_examples, validation_examples
