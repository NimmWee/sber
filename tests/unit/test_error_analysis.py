from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.error_analysis import analyze_prediction_errors
from eval.runner import RawLabeledExample


def test_analyze_prediction_errors_reports_requested_buckets() -> None:
    examples = [
        RawLabeledExample(
            prompt="Where is the conference?",
            response="The conference is in Paris on 2024-01-15 with 3 speakers.",
            label=0,
        ),
        RawLabeledExample(
            prompt="Who wrote Hamlet?",
            response="Napoleon wrote Hamlet.",
            label=1,
        ),
        RawLabeledExample(
            prompt="What is the capital of Italy?",
            response="Rome is the capital of Italy.",
            label=0,
        ),
        RawLabeledExample(
            prompt="Summarize the historical context in detail.",
            response=(
                "This answer includes several descriptive claims about Florence, "
                "Venice, and Milan across multiple historical eras and references."
            ),
            label=1,
        ),
    ]

    summary = analyze_prediction_errors(
        validation_examples=examples,
        probabilities=[0.92, 0.10, 0.20, 0.85],
        pr_auc=0.61,
    )

    assert summary.false_positive_count == 1
    assert summary.false_negative_count == 1
    assert "numbers" in summary.bucket_summaries
    assert "dates" in summary.bucket_summaries
    assert "entity_like_tokens" in summary.bucket_summaries
    assert "places" in summary.bucket_summaries
    assert "short_responses" in summary.bucket_summaries
    assert "long_responses" in summary.bucket_summaries
    assert summary.bucket_summaries["numbers"].total_count == 1
    assert summary.bucket_summaries["dates"].false_positive_count == 1
    assert summary.bucket_summaries["places"].false_positive_count == 1
    assert summary.bucket_summaries["entity_like_tokens"].total_count >= 3


def test_analyze_prediction_errors_sorts_hardest_examples_by_wrong_confidence() -> None:
    examples = [
        RawLabeledExample(
            prompt="What is the capital of Germany?",
            response="Munich is the capital of Germany.",
            label=1,
        ),
        RawLabeledExample(
            prompt="Who painted the Mona Lisa?",
            response="Leonardo da Vinci painted the Mona Lisa.",
            label=0,
        ),
        RawLabeledExample(
            prompt="How many moons does Mars have?",
            response="Mars has 12 moons.",
            label=1,
        ),
    ]

    summary = analyze_prediction_errors(
        validation_examples=examples,
        probabilities=[0.04, 0.88, 0.30],
        pr_auc=0.55,
    )

    assert summary.hardest_examples[0].response == "Munich is the capital of Germany."
    assert summary.hardest_examples[0].mistake_confidence > summary.hardest_examples[1].mistake_confidence
    assert set(summary.non_trivial_buckets)
