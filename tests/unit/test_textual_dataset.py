from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_dataset import (
    TextualTrainingExample,
    build_textual_training_dataset,
    load_public_seed_records,
    export_textual_training_dataset,
    load_textual_training_dataset,
)
from eval.runner import RawLabeledExample


def _write_seed_facts(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    '{"prompt":"Who discovered penicillin?",'
                    '"answer":"Alexander Fleming discovered penicillin in 1928.",'
                    '"source_name":"demo_facts",'
                    '"provenance":"https://example.test/facts#penicillin",'
                    '"metadata":{"bucket":"entity_like_tokens","entity_type":"person"}}'
                ),
                (
                    '{"prompt":"What is the capital of Peru?",'
                    '"answer":"Lima is the capital of Peru.",'
                    '"source_name":"demo_facts",'
                    '"provenance":"https://example.test/facts#peru",'
                    '"metadata":{"bucket":"places","entity_type":"place"}}'
                ),
                (
                    '{"prompt":"How many bones are in the adult human body?",'
                    '"answer":"The adult human body has 206 bones.",'
                    '"source_name":"demo_facts",'
                    '"provenance":"https://example.test/facts#bones",'
                    '"metadata":{"bucket":"numbers","entity_type":"quantity"}}'
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_build_textual_training_dataset_preserves_provenance_and_labels(tmp_path) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    _write_seed_facts(seed_path)

    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=[],
    )

    assert dataset.train_examples
    assert dataset.dev_examples
    assert {example.label for example in dataset.train_examples + dataset.dev_examples} == {0, 1}
    first_example = dataset.train_examples[0]
    assert isinstance(first_example, TextualTrainingExample)
    assert first_example.prompt
    assert first_example.response
    assert first_example.source_type in {"public_seed", "synthetic_corruption", "correct_augmentation"}
    assert first_example.source_name == "demo_facts"
    assert first_example.provenance.startswith("https://example.test/")
    assert first_example.generation_method
    assert "bucket_flags" in first_example.metadata
    assert "bucket" in first_example.metadata
    assert "entity_type" in first_example.metadata


def test_build_textual_training_dataset_reports_bucket_and_leakage_diagnostics(tmp_path) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    _write_seed_facts(seed_path)

    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=[
            RawLabeledExample(
                prompt="Who discovered penicillin?",
                response="Alexander Fleming discovered penicillin in 1928.",
                label=0,
            )
        ],
    )

    summary = dataset.summary
    assert summary["sample_size"] >= 12
    assert "hallucination_count" in summary
    assert "non_hallucination_count" in summary
    assert "source_distribution" in summary
    assert "source_name_distribution" in summary
    assert "source_type_distribution" in summary
    assert "bucket_coverage" in summary
    assert "bucket_label_counts" in summary
    assert summary["bucket_label_counts"]["numbers"]["hallucination_count"] >= 1
    assert summary["corruption_taxonomy"]["number_nearby"] >= 1
    assert "difficulty_heuristics" in summary
    assert "hallucination_bucket_coverage" in summary
    assert "leakage_checks" in summary
    assert summary["leakage_checks"]["public_exact_example_overlap_count"] >= 1
    assert any("source" in warning or "public benchmark" in warning for warning in summary["warnings"])


def test_build_textual_training_dataset_generates_diverse_correct_and_hallucinated_variants(
    tmp_path,
) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    _write_seed_facts(seed_path)

    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=[],
    )

    generation_methods = {
        example.generation_method
        for example in dataset.train_examples + dataset.dev_examples
    }
    corruption_types = {
        example.corruption_type
        for example in dataset.train_examples + dataset.dev_examples
        if example.corruption_type is not None
    }

    assert "source_seed_answer" in generation_methods
    assert "concise_restatement" in generation_methods
    assert "long_correct_variant" in generation_methods
    assert "nearby_number_corruption" in generation_methods or "nearby_year_corruption" in generation_methods
    assert {"number_nearby", "entity_swap"} & corruption_types


def test_build_textual_training_dataset_tracks_subtle_hallucination_difficulty(
    tmp_path,
) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    _write_seed_facts(seed_path)

    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=[],
    )

    difficulty = dataset.summary["difficulty_heuristics"]
    assert difficulty["small_numeric_delta_hallucination_count"] >= 1
    assert difficulty["same_type_entity_swap_count"] >= 1
    assert difficulty["long_local_corruption_count"] >= 0


def test_default_public_seed_corpus_has_stronger_risky_bucket_hallucination_coverage() -> None:
    dataset = build_textual_training_dataset(
        seed_path=PROJECT_ROOT / "data" / "public_seed_facts.jsonl",
        public_eval_examples=[],
    )

    hallucination_coverage = dataset.summary["hallucination_bucket_coverage"]

    assert hallucination_coverage["numbers"] >= 40
    assert hallucination_coverage["entity_like_tokens"] >= 30
    assert hallucination_coverage["long_responses"] >= 20


def test_export_and_load_textual_training_dataset_roundtrip(tmp_path) -> None:
    seed_path = tmp_path / "public_seed_facts.jsonl"
    export_path = tmp_path / "processed" / "textual_training_dataset.jsonl"
    _write_seed_facts(seed_path)

    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=[],
    )
    export_textual_training_dataset(dataset=dataset, output_path=export_path)
    loaded_examples = load_textual_training_dataset(export_path)

    assert export_path.exists()
    assert len(loaded_examples) == dataset.summary["sample_size"]
    assert loaded_examples[0].source_name == dataset.train_examples[0].source_name
    assert loaded_examples[0].generation_method == dataset.train_examples[0].generation_method


def test_load_public_seed_records_rejects_missing_required_fields(tmp_path) -> None:
    seed_path = tmp_path / "invalid_seed_facts.jsonl"
    seed_path.write_text(
        '{"prompt":"Who discovered penicillin?","answer":"Alexander Fleming discovered penicillin in 1928.","source_name":"demo"}',
        encoding="utf-8",
    )

    try:
        load_public_seed_records(seed_path)
    except ValueError as error:
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("expected ValueError for missing seed provenance field")


def test_default_public_seed_corpus_is_expanded_and_diverse() -> None:
    seed_path = PROJECT_ROOT / "data" / "public_seed_facts.jsonl"

    records = load_public_seed_records(seed_path)

    assert len(records) >= 20
    assert len({record.source_name for record in records}) >= 3
    assert sum(record.metadata.get("bucket") == "long_responses" for record in records) >= 4
    assert sum(record.metadata.get("bucket") == "numbers" for record in records) >= 4
