from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from utils.script_helpers import load_transformers_provider_config, write_json_artifact


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
