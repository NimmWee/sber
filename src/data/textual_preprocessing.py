from __future__ import annotations

from pathlib import Path

from data.textual_dataset import TextualTrainingExample, load_textual_training_dataset
from eval.default_detector import build_default_detector_extractor
from utils.script_helpers import (
    assert_not_public_benchmark_training_path,
    write_json_artifact,
)


def preprocess_textual_training_dataset(
    *,
    dataset_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
) -> dict[str, object]:
    assert_not_public_benchmark_training_path(dataset_path)
    examples = load_textual_training_dataset(dataset_path)
    extractor = build_default_detector_extractor()
    rows: list[dict[str, object]] = []
    feature_names: set[str] = set()

    for example in examples:
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=example.prompt,
            response=example.response,
        )
        features = dict(
            extractor.extract(
                prompt=example.prompt,
                response=example.response,
                token_stats=token_stats,
                internal_signal=internal_signal,
            )
        )
        feature_names.update(features.keys())
        rows.append(
            {
                "prompt": example.prompt,
                "response": example.response,
                "label": example.label,
                "split": example.split,
                "source_type": example.source_type,
                "source_name": example.source_name,
                "provenance": example.provenance,
                "generation_method": example.generation_method,
                "corruption_type": example.corruption_type,
                "metadata": example.metadata,
                "features": features,
            }
        )

    summary = {
        "sample_size": len(rows),
        "train_size": sum(row["split"] == "train" for row in rows),
        "dev_size": sum(row["split"] == "dev" for row in rows),
        "feature_names": sorted(feature_names),
    }
    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="preprocessed_text_training_dataset.json",
        payload={"summary": summary, "rows": rows},
    )
    return {
        "summary": summary,
        "rows": rows,
        "artifact_path": str(artifact_path),
    }


def train_detector_from_preprocessed_rows(
    *,
    preprocessed_rows: list[dict[str, object]],
    model_output_path: str | Path,
) -> dict[str, object]:
    from eval.default_detector import train_default_detector_head
    from eval.metrics import compute_pr_auc

    train_rows = [row for row in preprocessed_rows if row["split"] == "train"]
    dev_rows = [row for row in preprocessed_rows if row["split"] == "dev"]
    if not train_rows:
        raise ValueError("preprocessed_rows must contain train rows")
    if not dev_rows:
        raise ValueError("preprocessed_rows must contain dev rows")

    head = train_default_detector_head(
        feature_rows=[row["features"] for row in train_rows],
        labels=[int(row["label"]) for row in train_rows],
    )
    probabilities = head.predict_proba_batch([row["features"] for row in dev_rows])
    dev_pr_auc = compute_pr_auc([int(row["label"]) for row in dev_rows], probabilities)
    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    head.save(output_path)
    return {
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "dev_pr_auc": dev_pr_auc,
        "model_artifact_path": str(output_path),
    }


def score_private_dataset(
    *,
    input_path: str | Path,
    model_artifact_path: str | Path,
    token_stat_provider,
    output_path: str | Path,
) -> dict[str, object]:
    import csv

    from eval.default_detector import build_default_detector_extractor
    from models.head import TrainedLightGBMHead

    extractor = build_default_detector_extractor()
    head = TrainedLightGBMHead.load(model_artifact_path)
    input_csv_path = Path(input_path)
    output_csv_path = Path(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("private scoring input must not be empty")

    required = {"prompt"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("private scoring input must contain a prompt column")
    if "response" not in rows[0] and "model_answer" not in rows[0]:
        raise ValueError("private scoring input must contain response or model_answer column")

    output_rows: list[dict[str, object]] = []
    for row in rows:
        response = row.get("response") or row.get("model_answer") or ""
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=row["prompt"],
            response=response,
        )
        features = extractor.extract(
            prompt=row["prompt"],
            response=response,
            token_stats=token_stats,
            internal_signal=internal_signal,
        )
        output_rows.append(
            {
                **row,
                "hallucination_probability": head.predict_proba(features),
            }
        )

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "sample_size": len(output_rows),
        "output_path": str(output_csv_path),
    }


def _collect_signals(*, token_stat_provider, prompt: str, response: str):
    if hasattr(token_stat_provider, "collect_signals"):
        collected = token_stat_provider.collect_signals(prompt=prompt, response=response)
        return collected.token_stats, collected.internal_signal
    return token_stat_provider.collect(prompt=prompt, response=response), None
