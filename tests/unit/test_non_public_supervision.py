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
    assert dataset.summary["label_balance"]["hallucination_ratio"] < 0.5
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
    assert dataset.summary["risky_bucket_negative_coverage"]["numbers"] > 0
    assert dataset.summary["risky_bucket_negative_coverage"]["entity_like_tokens"] > 0
    assert dataset.summary["risky_bucket_negative_coverage"]["places"] > 0
    assert dataset.summary["risky_bucket_negative_coverage"]["long_responses"] > 0
    assert "near_duplicate_count" in dataset.summary
    assert "too_trivial_or_unrealistic_count" in dataset.summary
    assert "flagged_too_trivial_or_unrealistic_examples" in dataset.summary
    assert "warnings" in dataset.summary
    assert dataset.summary["leakage_checks"]["public_exact_example_overlap_count"] == 0
    assert dataset.summary["leakage_checks"]["public_prompt_overlap_count"] >= 0
