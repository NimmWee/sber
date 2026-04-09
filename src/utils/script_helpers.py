import json
from pathlib import Path

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

    bench_candidate = project_root_path / "data" / "bench" / "knowledge_bench_public.csv"
    if bench_candidate.exists():
        return bench_candidate

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
