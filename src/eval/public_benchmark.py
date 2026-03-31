from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.default_detector import build_default_detector_extractor
from eval.error_analysis import analyze_prediction_errors
from eval.metrics import compute_pr_auc
from eval.runner import RawLabeledExample
from models.head import TrainedLogisticRegressionHead
from utils.script_helpers import write_json_artifact


@dataclass(frozen=True)
class PublicBenchmarkEvaluationSummary:
    pr_auc: float
    sample_size: int
    false_positive_count: int
    false_negative_count: int
    non_trivial_buckets: list[str]
    summary_artifact_path: str
    per_example_artifact_path: str


def load_public_benchmark_examples(
    dataset_path: str | Path,
) -> list[RawLabeledExample]:
    csv_path = Path(dataset_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "prompt",
            "model_answer",
            "is_hallucination",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("required columns are missing from public benchmark CSV")

        examples: list[RawLabeledExample] = []
        for row in reader:
            examples.append(
                RawLabeledExample(
                    prompt=row["prompt"],
                    response=row["model_answer"],
                    label=_parse_label(row["is_hallucination"]),
                )
            )
    if not examples:
        raise ValueError("public benchmark CSV must not be empty")
    return examples


def evaluate_public_benchmark(
    *,
    dataset_path: str | Path,
    model_artifact_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
) -> PublicBenchmarkEvaluationSummary:
    examples = load_public_benchmark_examples(dataset_path)
    head = TrainedLogisticRegressionHead.load(model_artifact_path)
    extractor = build_default_detector_extractor()

    feature_rows = []
    probabilities = []
    per_example_rows = []
    for example in examples:
        token_stats = token_stat_provider.collect(
            prompt=example.prompt,
            response=example.response,
        )
        features = dict(
            extractor.extract(
                prompt=example.prompt,
                response=example.response,
                token_stats=token_stats,
            )
        )
        probability = head.predict_proba(features)
        feature_rows.append(features)
        probabilities.append(probability)
        per_example_rows.append(
            {
                "prompt": example.prompt,
                "response": example.response,
                "label": example.label,
                "probability": probability,
                "predicted_label": int(probability >= 0.5),
            }
        )

    labels = [example.label for example in examples]
    pr_auc = compute_pr_auc(labels, probabilities)
    error_summary = analyze_prediction_errors(
        validation_examples=examples,
        probabilities=probabilities,
        pr_auc=pr_auc,
    )

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_artifact_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="knowledge_bench_public_scores.json",
        payload={"rows": per_example_rows},
    )
    summary_artifact_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="knowledge_bench_public_summary.json",
        payload={
            "dataset_path": str(Path(dataset_path)),
            "model_artifact_path": str(Path(model_artifact_path)),
            "sample_size": len(examples),
            "pr_auc": pr_auc,
            "false_positive_count": error_summary.false_positive_count,
            "false_negative_count": error_summary.false_negative_count,
            "non_trivial_buckets": error_summary.non_trivial_buckets,
            "bucket_summaries": {
                name: asdict(summary)
                for name, summary in error_summary.bucket_summaries.items()
            },
            "hardest_examples": [asdict(example) for example in error_summary.hardest_examples],
            "per_example_artifact_path": str(per_example_artifact_path),
        },
    )

    return PublicBenchmarkEvaluationSummary(
        pr_auc=pr_auc,
        sample_size=len(examples),
        false_positive_count=error_summary.false_positive_count,
        false_negative_count=error_summary.false_negative_count,
        non_trivial_buckets=error_summary.non_trivial_buckets,
        summary_artifact_path=str(summary_artifact_path),
        per_example_artifact_path=str(per_example_artifact_path),
    )


def _parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return 1
    if normalized in {"false", "0", "no"}:
        return 0
    raise ValueError("is_hallucination must be a boolean-like value")
