from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from utils.script_helpers import (
    build_ablation_examples,
    load_frozen_submission_config,
    load_transformers_provider_config,
    resolve_frozen_submission_config,
    resolve_public_benchmark_path,
    resolve_triviaqa_path,
    resolve_text_training_seed_path,
    resolve_transformers_provider_config,
    write_json_artifact,
)


def test_load_transformers_provider_config_reads_model_path_and_delimiter(
    tmp_path,
) -> None:
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        (
            '{'
            '"model_id": "distilgpt2", '
            '"checkpoint_path": null, '
            '"device": "cpu", '
            '"torch_dtype": "auto", '
            '"response_delimiter": "|<resp>|"'
            '}'
        )
    )

    config = load_transformers_provider_config(config_path)

    assert config.model_id == "distilgpt2"
    assert config.response_delimiter == "|<resp>|"


def test_load_transformers_provider_config_defaults_device_to_auto(tmp_path) -> None:
    config_path = tmp_path / "provider_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16",
                "checkpoint_path": "/kaggle/temp/GigaChat3",
            }
        )
    )

    config = load_transformers_provider_config(config_path)

    assert config.device == "auto"


def test_write_json_artifact_creates_directory_and_writes_payload(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts" / "nested"

    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="summary.json",
        payload={"status": "ok", "score": 0.75},
    )

    assert artifact_path.exists()
    assert artifact_path.parent == artifact_dir
    assert artifact_path.read_text() == '{\n  "status": "ok",\n  "score": 0.75\n}'


def test_resolve_transformers_provider_config_prefers_explicit_path(tmp_path) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    checkpoint_dir = tmp_path / "gigachat"
    checkpoint_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    explicit_path = config_dir / "custom.json"
    explicit_path.write_text(
        json.dumps(
            {
                "model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16",
                "checkpoint_path": str(checkpoint_dir),
                "device": "cpu",
                "torch_dtype": "auto",
                "response_delimiter": "|<resp>|",
            }
        )
    )

    resolved = resolve_transformers_provider_config(
        project_root=project_root,
        explicit_config_path=explicit_path,
    )

    assert resolved.model_source == str(checkpoint_dir)


def test_resolve_transformers_provider_config_uses_local_override_when_checkpoint_exists(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    checkpoint_dir = tmp_path / "GigaChat3"
    checkpoint_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    (config_dir / "token_stat_provider.local.json").write_text(
        json.dumps(
            {
                "model_id": "distilgpt2",
                "checkpoint_path": str(checkpoint_dir),
                "device": "cpu",
                "torch_dtype": "auto",
                "response_delimiter": "|<resp>|",
            }
        )
    )
    (config_dir / "token_stat_provider.json").write_text(
        json.dumps({"model_id": "distilgpt2", "checkpoint_path": None})
    )

    resolved = resolve_transformers_provider_config(project_root=project_root)

    assert resolved.model_source == str(checkpoint_dir)


def test_resolve_transformers_provider_config_uses_kaggle_local_override_as_primary(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    checkpoint_dir = tmp_path / "kaggle" / "temp" / "GigaChat3"
    checkpoint_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    (config_dir / "token_stat_provider.local.json").write_text(
        json.dumps(
            {
                "model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16",
                "checkpoint_path": str(checkpoint_dir),
                "torch_dtype": "auto",
                "response_delimiter": "|<resp>|",
            }
        )
    )
    (config_dir / "token_stat_provider.json").write_text(
        json.dumps(
            {
                "model_id": "distilgpt2",
                "checkpoint_path": None,
            }
        )
    )

    resolved = resolve_transformers_provider_config(project_root=project_root)

    assert resolved.model_source == str(checkpoint_dir)
    assert resolved.device == "auto"


def test_resolve_transformers_provider_config_falls_back_when_local_override_is_not_a_real_checkpoint(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)

    (config_dir / "token_stat_provider.local.json").write_text(
        json.dumps({"model_id": "distilgpt2", "checkpoint_path": None})
    )
    (config_dir / "token_stat_provider.json").write_text(
        json.dumps(
            {
                "model_id": "ai-sage/GigaChat3-10B-A1.8B-bf16",
                "checkpoint_path": None,
                "device": "cpu",
                "torch_dtype": "auto",
                "response_delimiter": "|<resp>|",
            }
        )
    )

    resolved = resolve_transformers_provider_config(project_root=project_root)

    assert resolved.model_id == "ai-sage/GigaChat3-10B-A1.8B-bf16"


def test_build_ablation_examples_returns_larger_labeled_slice() -> None:
    train_examples, validation_examples = build_ablation_examples()

    assert len(train_examples) > 8
    assert len(validation_examples) > 4
    assert {example.label for example in train_examples} == {0, 1}
    assert {example.label for example in validation_examples} == {0, 1}


def test_resolve_public_benchmark_path_prefers_explicit_path(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    explicit_path = tmp_path / "custom.csv"
    explicit_path.write_text("prompt,model_answer,is_hallucination\n", encoding="utf-8")

    resolved = resolve_public_benchmark_path(
        project_root=project_root,
        explicit_dataset_path=explicit_path,
    )

    assert resolved == explicit_path


def test_resolve_public_benchmark_path_falls_back_to_project_root_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_path = project_root / "knowledge_bench_public.csv"
    dataset_path.write_text("prompt,model_answer,is_hallucination\n", encoding="utf-8")

    resolved = resolve_public_benchmark_path(project_root=project_root)

    assert resolved == dataset_path


def test_resolve_text_training_seed_path_prefers_explicit_path(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    explicit_path = tmp_path / "custom_seed_facts.jsonl"
    explicit_path.write_text("{}", encoding="utf-8")

    resolved = resolve_text_training_seed_path(
        project_root=project_root,
        explicit_seed_path=explicit_path,
    )

    assert resolved == explicit_path


def test_resolve_text_training_seed_path_uses_default_data_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)
    seed_path = data_dir / "public_seed_facts.jsonl"
    seed_path.write_text("{}", encoding="utf-8")

    resolved = resolve_text_training_seed_path(project_root=project_root)

    assert resolved == seed_path


def test_resolve_triviaqa_path_uses_local_data_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "triviaqa.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")

    resolved = resolve_triviaqa_path(project_root=project_root)

    assert resolved == dataset_path


def test_resolve_triviaqa_path_finds_pair_parquet_dataset(tmp_path) -> None:
    project_root = tmp_path / "project"
    pair_dir = project_root / "trivia-qa" / "pair"
    pair_dir.mkdir(parents=True)
    dataset_path = pair_dir / "train-00000-of-00001.parquet"
    dataset_path.write_text("", encoding="utf-8")

    resolved = resolve_triviaqa_path(project_root=project_root)

    assert resolved == dataset_path


def test_load_frozen_submission_config_reads_historical_best_metadata(tmp_path) -> None:
    config_path = tmp_path / "frozen_submission.json"
    config_path.write_text(
        json.dumps(
            {
                "historical_best_commit": "d3fa946",
                "historical_best_variant": "baseline_plus_all_specialists",
                "historical_best_pr_auc": 0.6881,
                "blend_weights": {
                    "baseline": 0.55,
                    "numeric": 0.15,
                    "entity": 0.15,
                    "long": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_frozen_submission_config(config_path)

    assert config["historical_best_commit"] == "d3fa946"
    assert config["historical_best_variant"] == "baseline_plus_all_specialists"
    assert config["blend_weights"]["baseline"] == 0.55


def test_resolve_frozen_submission_config_uses_default_config_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "frozen_submission.json"
    config_path.write_text(
        json.dumps(
            {
                "historical_best_commit": "d3fa946",
                "historical_best_variant": "baseline_plus_all_specialists",
                "historical_best_pr_auc": 0.6881,
                "blend_weights": {
                    "baseline": 0.55,
                    "numeric": 0.15,
                    "entity": 0.15,
                    "long": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_frozen_submission_config(project_root=project_root)

    assert resolved["historical_best_commit"] == "d3fa946"
