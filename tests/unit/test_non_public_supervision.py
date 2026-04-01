from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.non_public_supervision import build_non_public_supervision_dataset
from eval.runner import RawLabeledExample


def test_build_non_public_supervision_dataset_returns_balanced_taxonomy_and_leakage_summary() -> None:
    public_eval_examples = [
        RawLabeledExample(
            prompt="What is the capital of France?",
            response="Paris is the capital of France.",
            label=0,
        ),
        RawLabeledExample(
            prompt="Who wrote Hamlet?",
            response="Hamlet was written by Napoleon Bonaparte in 2024.",
            label=1,
        ),
    ]

    dataset = build_non_public_supervision_dataset(public_eval_examples=public_eval_examples)

    assert len(dataset.train_examples) > 0
    assert len(dataset.dev_examples) > 0
    assert dataset.summary["sample_size"] >= 300
    assert dataset.summary["train_size"] >= 240
    assert dataset.summary["dev_size"] >= 60
    assert dataset.summary["non_hallucination_count"] > dataset.summary["hallucination_count"]
    assert 0.35 <= dataset.summary["label_balance"]["hallucination_ratio"] <= 0.45
    assert "positive_count" not in dataset.summary
    assert "negative_count" not in dataset.summary
    assert dataset.summary["corruption_taxonomy"]["number_nearby"] >= 20
    assert dataset.summary["corruption_taxonomy"]["entity_swap"] >= 20
    assert dataset.summary["corruption_taxonomy"]["place_swap"] >= 20
    assert dataset.summary["corruption_taxonomy"]["organization_or_title_swap"] >= 20
    assert dataset.summary["corruption_taxonomy"]["date_nearby"] >= 20
    assert dataset.summary["long_response_count"] >= 40
    assert dataset.summary["number_heavy_count"] >= 40
    assert dataset.summary["entity_heavy_count"] >= 40
    assert dataset.summary["place_rich_count"] >= 20
    assert dataset.summary["approximate_or_range_style_count"] >= 20
    assert dataset.summary["bucket_label_counts"]["numbers"]["non_hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["numbers"]["hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["entity_like_tokens"]["non_hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["entity_like_tokens"]["hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["places"]["non_hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["places"]["hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["long_responses"]["non_hallucination_count"] > 0
    assert dataset.summary["bucket_label_counts"]["long_responses"]["hallucination_count"] > 0
    assert dataset.summary["bucket_label_ratios"]["numbers"]["hallucination_ratio"] > 0.2
    assert dataset.summary["bucket_label_ratios"]["long_responses"]["hallucination_ratio"] > 0.2
    assert dataset.summary["effective_label_balance"]["hallucination_ratio"] > 0.35
    assert dataset.summary["effective_label_balance"]["hallucination_ratio"] < 0.5
    assert "near_duplicate_count" in dataset.summary
    assert "too_trivial_or_unrealistic_count" in dataset.summary
    assert "flagged_too_trivial_or_unrealistic_examples" in dataset.summary
    assert "warnings" in dataset.summary
    assert dataset.summary["leakage_checks"]["public_exact_example_overlap_count"] == 0
    assert dataset.summary["leakage_checks"]["public_prompt_overlap_count"] >= 0


def test_build_non_public_supervision_dataset_keeps_label_semantics_explicit() -> None:
    dataset = build_non_public_supervision_dataset(public_eval_examples=[])

    hallucinated_train_count = sum(example.label for example in dataset.train_examples)
    hallucinated_dev_count = sum(example.label for example in dataset.dev_examples)
    assert hallucinated_train_count + hallucinated_dev_count == dataset.summary["hallucination_count"]
    assert (
        len(dataset.train_examples)
        + len(dataset.dev_examples)
        - hallucinated_train_count
        - hallucinated_dev_count
        == dataset.summary["non_hallucination_count"]
    )
    assert len(dataset.train_sample_weights) == len(dataset.train_examples)
    assert len(dataset.dev_sample_weights) == len(dataset.dev_examples)
    assert max(dataset.train_sample_weights) > min(dataset.train_sample_weights)
