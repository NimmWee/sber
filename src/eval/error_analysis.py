import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.default_detector import filter_default_detector_rows, train_default_detector_head
from eval.metrics import compute_pr_auc
from eval.runner import RawExampleEvaluationDataset, RawLabeledExample


SHORT_RESPONSE_TOKEN_THRESHOLD = 8
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z-]+\b")
LOCATION_PREPOSITION_PATTERN = re.compile(
    r"\b(?:in|at|from|to|near|around)\s+[A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*\b"
)


@dataclass(frozen=True)
class ErrorExampleSummary:
    prompt: str
    response: str
    label: int
    predicted_label: int
    probability: float
    mistake_confidence: float
    buckets: list[str]


@dataclass(frozen=True)
class ErrorBucketSummary:
    total_count: int
    false_positive_count: int
    false_negative_count: int


@dataclass(frozen=True)
class ErrorAnalysisSummary:
    pr_auc: float
    sample_size: int
    false_positive_count: int
    false_negative_count: int
    hardest_examples: list[ErrorExampleSummary]
    bucket_summaries: dict[str, ErrorBucketSummary]
    focused_bucket_summaries: dict[str, ErrorBucketSummary]
    non_trivial_buckets: list[str]
    recommended_next_improvement: str
    model_artifact_path: str | None = None
    summary_artifact_path: str | None = None


def analyze_prediction_errors(
    *,
    validation_examples: list[RawLabeledExample],
    probabilities: list[float],
    pr_auc: float,
    threshold: float = 0.5,
    max_hardest_examples: int = 5,
) -> ErrorAnalysisSummary:
    if len(validation_examples) != len(probabilities):
        raise ValueError("validation_examples and probabilities must have the same length")

    bucket_names = (
        "numbers",
        "dates",
        "entity_like_tokens",
        "places",
        "short_responses",
        "long_responses",
    )
    bucket_totals = {name: 0 for name in bucket_names}
    bucket_false_positives = {name: 0 for name in bucket_names}
    bucket_false_negatives = {name: 0 for name in bucket_names}

    hardest_examples: list[ErrorExampleSummary] = []
    false_positive_count = 0
    false_negative_count = 0

    for example, probability in zip(validation_examples, probabilities):
        predicted_label = 1 if probability >= threshold else 0
        buckets = _example_buckets(example)

        for bucket in buckets:
            bucket_totals[bucket] += 1

        if predicted_label == 1 and example.label == 0:
            false_positive_count += 1
            for bucket in buckets:
                bucket_false_positives[bucket] += 1
            hardest_examples.append(
                ErrorExampleSummary(
                    prompt=example.prompt,
                    response=example.response,
                    label=example.label,
                    predicted_label=predicted_label,
                    probability=float(probability),
                    mistake_confidence=float(probability),
                    buckets=buckets,
                )
            )
        elif predicted_label == 0 and example.label == 1:
            false_negative_count += 1
            for bucket in buckets:
                bucket_false_negatives[bucket] += 1
            hardest_examples.append(
                ErrorExampleSummary(
                    prompt=example.prompt,
                    response=example.response,
                    label=example.label,
                    predicted_label=predicted_label,
                    probability=float(probability),
                    mistake_confidence=float(1.0 - probability),
                    buckets=buckets,
                )
            )

    hardest_examples = sorted(
        hardest_examples,
        key=lambda example: example.mistake_confidence,
        reverse=True,
    )[:max_hardest_examples]

    bucket_summaries = {
        bucket_name: ErrorBucketSummary(
            total_count=bucket_totals[bucket_name],
            false_positive_count=bucket_false_positives[bucket_name],
            false_negative_count=bucket_false_negatives[bucket_name],
        )
        for bucket_name in bucket_names
    }
    non_trivial_buckets = [
        bucket_name
        for bucket_name, summary in bucket_summaries.items()
        if summary.false_positive_count + summary.false_negative_count > 0
    ]
    focused_bucket_names = (
        "numbers",
        "entity_like_tokens",
        "places",
        "short_responses",
        "long_responses",
    )
    focused_bucket_summaries = {
        bucket_name: bucket_summaries[bucket_name]
        for bucket_name in focused_bucket_names
    }

    return ErrorAnalysisSummary(
        pr_auc=float(pr_auc),
        sample_size=len(validation_examples),
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        hardest_examples=hardest_examples,
        bucket_summaries=bucket_summaries,
        focused_bucket_summaries=focused_bucket_summaries,
        non_trivial_buckets=non_trivial_buckets,
        recommended_next_improvement=_recommend_next_improvement(
            focused_bucket_summaries
        ),
    )


class DefaultDetectorErrorAnalysisRunner:
    def __init__(
        self,
        *,
        dataset: RawExampleEvaluationDataset,
        artifact_dir: str | Path,
    ) -> None:
        self.dataset = dataset
        self.artifact_dir = Path(artifact_dir)

    def run(self) -> ErrorAnalysisSummary:
        split = self.dataset.load_split()
        validation_features = filter_default_detector_rows(
            feature_rows=split.validation_features,
        )

        model = train_default_detector_head(
            feature_rows=split.train_features,
            labels=split.train_labels,
        )
        probabilities = model.predict_proba_batch(validation_features)
        pr_auc = compute_pr_auc(split.validation_labels, probabilities)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        model_artifact_path = self.artifact_dir / "default_detector_head.json"
        summary_artifact_path = self.artifact_dir / "error_analysis_summary.json"
        model.save(model_artifact_path)

        summary = analyze_prediction_errors(
            validation_examples=self.dataset.validation_examples,
            probabilities=probabilities,
            pr_auc=pr_auc,
        )
        summary = ErrorAnalysisSummary(
            pr_auc=summary.pr_auc,
            sample_size=summary.sample_size,
            false_positive_count=summary.false_positive_count,
            false_negative_count=summary.false_negative_count,
            hardest_examples=summary.hardest_examples,
            bucket_summaries=summary.bucket_summaries,
            focused_bucket_summaries=summary.focused_bucket_summaries,
            non_trivial_buckets=summary.non_trivial_buckets,
            recommended_next_improvement=summary.recommended_next_improvement,
            model_artifact_path=str(model_artifact_path),
            summary_artifact_path=str(summary_artifact_path),
        )
        summary_artifact_path.write_text(json.dumps(asdict(summary), indent=2))
        return summary


def _example_buckets(example: RawLabeledExample) -> list[str]:
    response = example.response
    response_tokens = re.findall(r"\w+", response)
    buckets: list[str] = []

    if any(character.isdigit() for character in response):
        buckets.append("numbers")
    if DATE_PATTERN.search(response):
        buckets.append("dates")
    if ENTITY_PATTERN.search(response):
        buckets.append("entity_like_tokens")
    if _has_location_like_mentions(prompt=example.prompt, response=response):
        buckets.append("places")
    if len(response_tokens) <= SHORT_RESPONSE_TOKEN_THRESHOLD:
        buckets.append("short_responses")
    else:
        buckets.append("long_responses")

    return buckets


def _has_location_like_mentions(*, prompt: str, response: str) -> bool:
    if LOCATION_PREPOSITION_PATTERN.search(response):
        return True

    prompt_lower = prompt.lower()
    location_prompt_cues = ("capital", "city", "country", "located", "location", "place")
    if any(cue in prompt_lower for cue in location_prompt_cues):
        return bool(ENTITY_PATTERN.search(response))
    return False


def _recommend_next_improvement(
    focused_bucket_summaries: dict[str, ErrorBucketSummary],
) -> str:
    prioritized_buckets = (
        "entity_like_tokens",
        "places",
        "numbers",
        "short_responses",
        "long_responses",
    )
    dominant_bucket = max(
        prioritized_buckets,
        key=lambda bucket_name: (
            focused_bucket_summaries[bucket_name].false_positive_count
            + focused_bucket_summaries[bucket_name].false_negative_count
        ),
    )
    if (
        focused_bucket_summaries[dominant_bucket].false_positive_count
        + focused_bucket_summaries[dominant_bucket].false_negative_count
        == 0
    ):
        return "No model change yet; expand evaluation before adding new features."
    if dominant_bucket in {"entity_like_tokens", "places"}:
        return "Expand non-public entity and place supervision before adding new model features."
    if dominant_bucket == "numbers":
        return "Add more non-public numeric contradiction examples before new feature work."
    if dominant_bucket == "short_responses":
        return "Improve short-answer coverage in non-public training data before changing the model."
    return "Expand non-public long-response supervision before adding feature complexity."
