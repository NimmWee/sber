from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_dataset import build_textual_training_dataset, export_textual_training_dataset
from data.textual_preprocessing import preprocess_textual_training_dataset
from features.extractor import InternalModelSignal, TokenUncertaintyStat
from inference.token_stats import CollectedModelSignals


class DeterministicSignalProvider:
    def collect_signals(self, prompt: str, response: str) -> CollectedModelSignals:
        is_suspicious = "wrong" in response.lower() or any(character.isdigit() for character in response)
        return CollectedModelSignals(
            token_stats=[
                TokenUncertaintyStat(
                    token="fact",
                    logprob=-1.2 if is_suspicious else -0.1,
                    entropy=0.9 if is_suspicious else 0.2,
                    top1_top2_margin=0.1 if is_suspicious else 0.7,
                ),
                TokenUncertaintyStat(
                    token="entity",
                    logprob=-0.9 if is_suspicious else -0.15,
                    entropy=0.8 if is_suspicious else 0.25,
                    top1_top2_margin=0.2 if is_suspicious else 0.6,
                ),
            ],
            internal_signal=InternalModelSignal(
                last_layer_pooled_l2=1.3,
                last_layer_pooled_mean_abs=0.4,
                selected_layer_norm_variance=0.09,
                layer_disagreement_mean=0.14,
                selected_layer_disagreement_max=0.2,
                early_late_layer_consistency=0.81,
            ),
        )


def _write_seed_facts(path: Path) -> None:
    path.write_text(
        (
            '{"prompt":"Who discovered penicillin?",'
            '"answer":"Alexander Fleming discovered penicillin in 1928.",'
            '"source_name":"demo_facts",'
            '"provenance":"https://example.test/facts#penicillin",'
            '"metadata":{"bucket":"entity_like_tokens","entity_type":"person"}}\n'
            '{"prompt":"What is the capital of Peru?",'
            '"answer":"Lima is the capital of Peru.",'
            '"source_name":"demo_facts",'
            '"provenance":"https://example.test/facts#peru",'
            '"metadata":{"bucket":"places","entity_type":"place"}}'
        ),
        encoding="utf-8",
    )


def test_preprocess_textual_training_dataset_is_reproducible(tmp_path) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    dataset_path = tmp_path / "processed" / "textual_training_dataset.jsonl"
    _write_seed_facts(seed_path)
    dataset = build_textual_training_dataset(seed_path=seed_path, public_eval_examples=[])
    export_textual_training_dataset(dataset=dataset, output_path=dataset_path)

    first = preprocess_textual_training_dataset(
        dataset_path=dataset_path,
        token_stat_provider=DeterministicSignalProvider(),
        artifact_dir=tmp_path / "artifacts" / "first",
    )
    second = preprocess_textual_training_dataset(
        dataset_path=dataset_path,
        token_stat_provider=DeterministicSignalProvider(),
        artifact_dir=tmp_path / "artifacts" / "second",
    )

    assert first["summary"]["sample_size"] == second["summary"]["sample_size"]
    assert first["summary"]["feature_names"] == second["summary"]["feature_names"]
    assert first["rows"] == second["rows"]
    assert Path(first["artifact_path"]).exists()
    assert Path(second["artifact_path"]).exists()
