from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data.textual_dataset import load_textual_training_dataset
from eval.default_detector import DEFAULT_TOKEN_FEATURE_GROUPS
from eval.metrics import compute_pr_auc
from features.extractor import InternalModelSignal, StructuralFeatureExtractor, TokenUncertaintyStat
from models.head import TrainedLightGBMHead, train_lightgbm_head
from utils.script_helpers import (
    assert_not_public_benchmark_training_path,
    assert_unlabeled_submission_input_columns,
    build_runtime_metadata,
    write_json_artifact,
    write_markdown_artifact,
)


HISTORICAL_BEST_COMMIT = "d3fa946"
HISTORICAL_BEST_VARIANT = "baseline_plus_all_specialists"
HISTORICAL_BEST_PR_AUC = 0.6881
FROZEN_BLEND_WEIGHTS = {
    "baseline": 0.55,
    "numeric": 0.15,
    "entity": 0.15,
    "long": 0.15,
}
FROZEN_BLEND_VERSION = "baseline_plus_all_specialists_v1"
FROZEN_BLEND_WEIGHT_SOURCE = (
    "frozen historical submission blend, preserved for reproducibility and not re-tuned in this repo"
)
DEFAULT_SERVING_THRESHOLD = 0.3
SERVING_THRESHOLD_SOURCE = (
    "fixed serving threshold for optional boolean mode; not used for PR-AUC optimization"
)


@dataclass(frozen=True)
class FrozenSubmissionBundle:
    baseline_head: Any
    numeric_head: Any
    entity_head: Any
    long_head: Any
    metadata: dict[str, Any]


def build_frozen_best_metadata(*, project_root: str | Path | None = None) -> dict[str, Any]:
    metadata = {
        "historical_best_commit": HISTORICAL_BEST_COMMIT,
        "historical_best_variant": HISTORICAL_BEST_VARIANT,
        "historical_best_pr_auc": HISTORICAL_BEST_PR_AUC,
        "blend_weights": dict(FROZEN_BLEND_WEIGHTS),
        "blend_version": FROZEN_BLEND_VERSION,
        "blend_weight_source": FROZEN_BLEND_WEIGHT_SOURCE,
        "primary_output_mode": "probability",
        "serving_threshold": DEFAULT_SERVING_THRESHOLD,
        "serving_threshold_source": SERVING_THRESHOLD_SOURCE,
        "random_seed": 0,
    }
    if project_root is not None:
        metadata["runtime"] = build_runtime_metadata(project_root=project_root)
    return metadata


def train_frozen_best_submission(
    *,
    dataset_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    assert_not_public_benchmark_training_path(dataset_path)
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(
            "textual training dataset was not found. "
            "Run `python scripts/build_text_training_dataset.py` first or pass --dataset-path."
        )
    examples = load_textual_training_dataset(dataset_path)
    train_examples = [example for example in examples if example.split == "train"]
    dev_examples = [example for example in examples if example.split == "dev"]
    if not train_examples:
        raise ValueError("frozen submission training dataset must contain train examples")
    if not dev_examples:
        raise ValueError("frozen submission training dataset must contain dev examples")

    extractor = _build_frozen_best_extractor()
    train_base_rows, train_specialist_rows, train_labels = _prepare_rows(
        examples=train_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
    )
    dev_base_rows, dev_specialist_rows, dev_labels = _prepare_rows(
        examples=dev_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
    )

    baseline_head = train_lightgbm_head(train_base_rows, train_labels)
    numeric_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_numeric_",
        ),
        train_labels,
    )
    entity_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_entity_",
        ),
        train_labels,
    )
    long_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_long_",
        ),
        train_labels,
    )

    dev_probabilities = _predict_frozen_best_probabilities(
        base_rows=dev_base_rows,
        specialist_rows=dev_specialist_rows,
        bundle=FrozenSubmissionBundle(
            baseline_head=baseline_head,
            numeric_head=numeric_head,
            entity_head=entity_head,
            long_head=long_head,
            metadata=build_frozen_best_metadata(),
        ),
    )
    dev_pr_auc = compute_pr_auc(dev_labels, dev_probabilities)

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_head.save(output_dir / "baseline_head.json")
    numeric_head.save(output_dir / "numeric_specialist_head.json")
    entity_head.save(output_dir / "entity_specialist_head.json")
    long_head.save(output_dir / "long_specialist_head.json")
    metadata = {
        **build_frozen_best_metadata(project_root=project_root),
        "train_size": len(train_examples),
        "dev_size": len(dev_examples),
        "dev_pr_auc": dev_pr_auc,
    }
    metadata_path = write_json_artifact(
        artifact_dir=output_dir,
        filename="frozen_submission_metadata.json",
        payload=metadata,
    )
    return {
        "train_size": len(train_examples),
        "dev_size": len(dev_examples),
        "dev_pr_auc": dev_pr_auc,
        "artifact_dir": str(output_dir),
        "metadata_path": str(metadata_path),
    }


def load_frozen_submission_bundle(artifact_dir: str | Path) -> FrozenSubmissionBundle:
    artifact_root = Path(artifact_dir)
    required_artifacts = [
        artifact_root / "baseline_head.json",
        artifact_root / "numeric_specialist_head.json",
        artifact_root / "entity_specialist_head.json",
        artifact_root / "long_specialist_head.json",
    ]
    missing_artifacts = [str(path) for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError(
            "frozen submission artifacts were not found. "
            "Run `bash scripts/train.sh --config configs/token_stat_provider.local.json` first. "
            f"Missing: {', '.join(missing_artifacts)}"
        )
    metadata = build_frozen_best_metadata()
    metadata_path = artifact_root / "frozen_submission_metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return FrozenSubmissionBundle(
        baseline_head=TrainedLightGBMHead.load(artifact_root / "baseline_head.json"),
        numeric_head=TrainedLightGBMHead.load(artifact_root / "numeric_specialist_head.json"),
        entity_head=TrainedLightGBMHead.load(artifact_root / "entity_specialist_head.json"),
        long_head=TrainedLightGBMHead.load(artifact_root / "long_specialist_head.json"),
        metadata=metadata,
    )


def score_private_frozen_submission(
    *,
    input_path: str | Path,
    output_path: str | Path,
    token_stat_provider,
    bundle: FrozenSubmissionBundle | None = None,
    artifact_dir: str | Path | None = None,
    output_mode: str = "probability",
    label_threshold: float = DEFAULT_SERVING_THRESHOLD,
) -> dict[str, Any]:
    input_csv_path = Path(input_path)
    if not input_csv_path.exists():
        raise FileNotFoundError(
            "knowledge_bench_private.csv was not found. "
            "Place the private benchmark at data/bench/knowledge_bench_private.csv "
            "or pass --input-path explicitly."
        )
    if bundle is None:
        if artifact_dir is None:
            raise ValueError("artifact_dir is required when bundle is not provided")
        bundle = load_frozen_submission_bundle(artifact_dir)
    if output_mode not in {"probability", "boolean"}:
        raise ValueError("output_mode must be either 'probability' or 'boolean'")

    extractor = _build_frozen_best_extractor()
    output_csv_path = Path(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert_unlabeled_submission_input_columns(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError("private scoring input must not be empty")
    if "prompt" not in rows[0]:
        raise ValueError("private scoring input must contain a prompt column")
    if "response" not in rows[0] and "model_answer" not in rows[0]:
        raise ValueError("private scoring input must contain response or model_answer column")

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        response = row.get("response") or row.get("model_answer") or ""
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=row["prompt"],
            response=response,
        )
        base_row = dict(
            extractor.extract(
                prompt=row["prompt"],
                response=response,
                token_stats=token_stats,
                internal_signal=internal_signal,
            )
        )
        specialist_row = _extract_historical_specialist_features(
            prompt=row["prompt"],
            response=response,
            token_stats=token_stats,
            internal_signal=internal_signal,
        )
        probability = _blend_probabilities(
            baseline_score=bundle.baseline_head.predict_proba(base_row),
            numeric_score=bundle.numeric_head.predict_proba(
                _select_historical_specialist_feature_row(
                    specialist_row=specialist_row,
                    prefix="specialist_numeric_",
                )
            ),
            entity_score=bundle.entity_head.predict_proba(
                _select_historical_specialist_feature_row(
                    specialist_row=specialist_row,
                    prefix="specialist_entity_",
                )
            ),
            long_score=bundle.long_head.predict_proba(
                _select_historical_specialist_feature_row(
                    specialist_row=specialist_row,
                    prefix="specialist_long_",
                )
            ),
        )
        scored_rows.append(
            _format_scored_row(
                row=row,
                probability=probability,
                output_mode=output_mode,
                label_threshold=label_threshold,
            )
        )

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)
    return {
        "sample_size": len(scored_rows),
        "output_path": str(output_csv_path),
        "metadata": bundle.metadata,
    }


def run_internal_ablation_report(
    *,
    dataset_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
    report_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    assert_not_public_benchmark_training_path(dataset_path)
    examples = load_textual_training_dataset(dataset_path)
    train_examples = [example for example in examples if example.split == "train"]
    dev_examples = [example for example in examples if example.split == "dev"]
    if not train_examples or not dev_examples:
        raise ValueError("internal ablation requires both train and dev splits")

    extractor = _build_frozen_best_extractor()
    train_base_rows, train_specialist_rows, train_labels = _prepare_rows(
        examples=train_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
    )
    dev_base_rows, dev_specialist_rows, dev_labels = _prepare_rows(
        examples=dev_examples,
        extractor=extractor,
        token_stat_provider=token_stat_provider,
    )

    baseline_head = train_lightgbm_head(train_base_rows, train_labels)
    numeric_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_numeric_",
        ),
        train_labels,
    )
    entity_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_entity_",
        ),
        train_labels,
    )
    long_head = train_lightgbm_head(
        _select_historical_specialist_feature_subset(
            specialist_rows=train_specialist_rows,
            prefix="specialist_long_",
        ),
        train_labels,
    )

    baseline_scores = baseline_head.predict_proba_batch(dev_base_rows)
    numeric_scores = numeric_head.predict_proba_batch(
        _select_historical_specialist_feature_subset(
            specialist_rows=dev_specialist_rows,
            prefix="specialist_numeric_",
        )
    )
    entity_scores = entity_head.predict_proba_batch(
        _select_historical_specialist_feature_subset(
            specialist_rows=dev_specialist_rows,
            prefix="specialist_entity_",
        )
    )
    long_scores = long_head.predict_proba_batch(
        _select_historical_specialist_feature_subset(
            specialist_rows=dev_specialist_rows,
            prefix="specialist_long_",
        )
    )
    full_blend_scores = [
        _blend_probabilities(
            baseline_score=baseline_score,
            numeric_score=numeric_score,
            entity_score=entity_score,
            long_score=long_score,
        )
        for baseline_score, numeric_score, entity_score, long_score in zip(
            baseline_scores,
            numeric_scores,
            entity_scores,
            long_scores,
        )
    ]

    variants = {
        "baseline_only": _internal_variant_summary(dev_labels, baseline_scores),
        "baseline_plus_numeric": _internal_variant_summary(
            dev_labels,
            [
                (FROZEN_BLEND_WEIGHTS["baseline"] * baseline)
                + (FROZEN_BLEND_WEIGHTS["numeric"] * numeric)
                for baseline, numeric in zip(baseline_scores, numeric_scores)
            ],
        ),
        "baseline_plus_entity": _internal_variant_summary(
            dev_labels,
            [
                (FROZEN_BLEND_WEIGHTS["baseline"] * baseline)
                + (FROZEN_BLEND_WEIGHTS["entity"] * entity)
                for baseline, entity in zip(baseline_scores, entity_scores)
            ],
        ),
        "baseline_plus_long": _internal_variant_summary(
            dev_labels,
            [
                (FROZEN_BLEND_WEIGHTS["baseline"] * baseline)
                + (FROZEN_BLEND_WEIGHTS["long"] * long_score)
                for baseline, long_score in zip(baseline_scores, long_scores)
            ],
        ),
        "full_blend": _internal_variant_summary(dev_labels, full_blend_scores),
    }
    best_variant = max(variants, key=lambda name: variants[name]["pr_auc"])
    payload = {
        "dataset_path": str(dataset_path),
        "report_scope": "internal_validation_only",
        "best_variant": best_variant,
        "blend_metadata": build_frozen_best_metadata(project_root=project_root),
        "variants": variants,
    }
    report_root = Path(report_dir)
    json_report_path = write_json_artifact(
        artifact_dir=report_root,
        filename="ablation_report.json",
        payload=payload,
    )
    markdown_report_path = write_markdown_artifact(
        artifact_dir=report_root,
        filename="ablation_report.md",
        markdown=_build_ablation_markdown(payload),
    )
    return {
        **payload,
        "json_report_path": str(json_report_path),
        "markdown_report_path": str(markdown_report_path),
    }


def benchmark_frozen_submission_latency(
    *,
    dataset_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
    report_dir: str | Path,
    sample_size: int,
) -> dict[str, Any]:
    bundle = load_frozen_submission_bundle(artifact_dir)
    rows = _load_prompt_response_rows(dataset_path)
    if not rows:
        raise ValueError("latency benchmark dataset must not be empty")
    selected_rows = rows[:sample_size]
    extractor = _build_frozen_best_extractor()

    provider_samples: list[float] = []
    feature_samples: list[float] = []
    specialist_samples: list[float] = []
    blend_samples: list[float] = []
    total_samples: list[float] = []

    for row in selected_rows:
        prompt = row["prompt"]
        response = row["response"]

        provider_start = __import__("time").perf_counter()
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=prompt,
            response=response,
        )
        provider_samples.append((__import__("time").perf_counter() - provider_start) * 1000.0)

        feature_start = __import__("time").perf_counter()
        base_row = dict(
            extractor.extract(
                prompt=prompt,
                response=response,
                token_stats=token_stats,
                internal_signal=internal_signal,
            )
        )
        specialist_row = _extract_historical_specialist_features(
            prompt=prompt,
            response=response,
            token_stats=token_stats,
            internal_signal=internal_signal,
        )
        feature_samples.append((__import__("time").perf_counter() - feature_start) * 1000.0)

        specialist_start = __import__("time").perf_counter()
        baseline_score = bundle.baseline_head.predict_proba(base_row)
        numeric_score = bundle.numeric_head.predict_proba(
            _select_historical_specialist_feature_row(
                specialist_row=specialist_row,
                prefix="specialist_numeric_",
            )
        )
        entity_score = bundle.entity_head.predict_proba(
            _select_historical_specialist_feature_row(
                specialist_row=specialist_row,
                prefix="specialist_entity_",
            )
        )
        long_score = bundle.long_head.predict_proba(
            _select_historical_specialist_feature_row(
                specialist_row=specialist_row,
                prefix="specialist_long_",
            )
        )
        specialist_samples.append((__import__("time").perf_counter() - specialist_start) * 1000.0)

        blend_start = __import__("time").perf_counter()
        _ = _blend_probabilities(
            baseline_score=baseline_score,
            numeric_score=numeric_score,
            entity_score=entity_score,
            long_score=long_score,
        )
        blend_samples.append((__import__("time").perf_counter() - blend_start) * 1000.0)

        total_samples.append(
            provider_samples[-1]
            + feature_samples[-1]
            + specialist_samples[-1]
            + blend_samples[-1]
        )

    payload = {
        "sample_size": len(selected_rows),
        "provider_forward_ms": _latency_stats(provider_samples),
        "feature_extraction_ms": _latency_stats(feature_samples),
        "specialist_scoring_ms": _latency_stats(specialist_samples),
        "blend_and_formatting_ms": _latency_stats(blend_samples),
        "total_ms": _latency_stats(total_samples),
        "metadata": bundle.metadata,
    }
    report_root = Path(report_dir)
    json_report_path = write_json_artifact(
        artifact_dir=report_root,
        filename="latency_report.json",
        payload=payload,
    )
    markdown_report_path = write_markdown_artifact(
        artifact_dir=report_root,
        filename="latency_report.md",
        markdown=_build_latency_markdown(payload),
    )
    return {
        **payload,
        "json_report_path": str(json_report_path),
        "markdown_report_path": str(markdown_report_path),
    }


def _format_scored_row(
    *,
    row: Mapping[str, Any],
    probability: float,
    output_mode: str,
    label_threshold: float,
) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    response = str(row.get("response") or row.get("model_answer") or "")
    if output_mode == "boolean":
        return {
            "prompt": prompt,
            "response": response,
            "hallucination": "true" if probability >= label_threshold else "false",
        }
    return {
        "prompt": prompt,
        "response": response,
        "hallucination_probability": probability,
    }


def _build_frozen_best_extractor() -> StructuralFeatureExtractor:
    return StructuralFeatureExtractor(
        enable_token_uncertainty=True,
        enable_internal_features=True,
        enable_compact_internal_enhancements=True,
        token_feature_groups=DEFAULT_TOKEN_FEATURE_GROUPS,
    )


def _prepare_rows(
    *,
    examples,
    extractor: StructuralFeatureExtractor,
    token_stat_provider,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[int]]:
    base_rows: list[dict[str, float]] = []
    specialist_rows: list[dict[str, float]] = []
    labels: list[int] = []
    for example in examples:
        token_stats, internal_signal = _collect_signals(
            token_stat_provider=token_stat_provider,
            prompt=example.prompt,
            response=example.response,
        )
        base_rows.append(
            dict(
                extractor.extract(
                    prompt=example.prompt,
                    response=example.response,
                    token_stats=token_stats,
                    internal_signal=internal_signal,
                )
            )
        )
        specialist_rows.append(
            _extract_historical_specialist_features(
                prompt=example.prompt,
                response=example.response,
                token_stats=token_stats,
                internal_signal=internal_signal,
            )
        )
        labels.append(int(example.label))
    return base_rows, specialist_rows, labels


def _predict_frozen_best_probabilities(
    *,
    base_rows: list[Mapping[str, float]],
    specialist_rows: list[dict[str, float]],
    bundle: FrozenSubmissionBundle,
) -> list[float]:
    probabilities: list[float] = []
    for base_row, specialist_row in zip(base_rows, specialist_rows):
        probabilities.append(
            _blend_probabilities(
                baseline_score=bundle.baseline_head.predict_proba(base_row),
                numeric_score=bundle.numeric_head.predict_proba(
                    _select_historical_specialist_feature_row(
                        specialist_row=specialist_row,
                        prefix="specialist_numeric_",
                    )
                ),
                entity_score=bundle.entity_head.predict_proba(
                    _select_historical_specialist_feature_row(
                        specialist_row=specialist_row,
                        prefix="specialist_entity_",
                    )
                ),
                long_score=bundle.long_head.predict_proba(
                    _select_historical_specialist_feature_row(
                        specialist_row=specialist_row,
                        prefix="specialist_long_",
                    )
                ),
            )
        )
    return probabilities


def _blend_probabilities(
    *,
    baseline_score: float,
    numeric_score: float,
    entity_score: float,
    long_score: float,
) -> float:
    return (
        (FROZEN_BLEND_WEIGHTS["baseline"] * baseline_score)
        + (FROZEN_BLEND_WEIGHTS["numeric"] * numeric_score)
        + (FROZEN_BLEND_WEIGHTS["entity"] * entity_score)
        + (FROZEN_BLEND_WEIGHTS["long"] * long_score)
    )


def _internal_variant_summary(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    metrics = _binary_metrics(labels, probabilities)
    return {
        "pr_auc": compute_pr_auc(labels, probabilities),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "predicted_positive_rate": metrics["predicted_positive_rate"],
    }


def _load_prompt_response_rows(path: str | Path) -> list[dict[str, str]]:
    input_csv_path = Path(path)
    if not input_csv_path.exists():
        raise FileNotFoundError(f"dataset was not found: {input_csv_path}")
    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert_unlabeled_submission_input_columns(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError("dataset must not be empty")
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response") or row.get("model_answer") or "")
        if not prompt:
            raise ValueError("dataset must contain a prompt column")
        if not response:
            raise ValueError("dataset must contain response or model_answer values")
        normalized_rows.append({"prompt": prompt, "response": response})
    return normalized_rows


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "avg": _mean(ordered),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
    }


def _build_ablation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ablation report",
        "",
        f"- scope: {payload['report_scope']}",
        f"- best_variant: {payload['best_variant']}",
        "",
        "| Variant | PR-AUC | Precision | Recall | Predicted positive rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in payload["variants"].items():
        lines.append(
            f"| {name} | {summary['pr_auc']:.4f} | {summary['precision']:.4f} | "
            f"{summary['recall']:.4f} | {summary['predicted_positive_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Short summary",
            "",
            "- This report uses only the internal train/dev split from the text-based training dataset.",
            "- Public benchmark is not used for fit or weight selection here.",
            "- Specialists that do not improve internal validation should be treated as optional in future iterations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_latency_markdown(payload: dict[str, Any]) -> str:
    return (
        "# Latency report\n\n"
        f"- sample_size: {payload['sample_size']}\n"
        f"- avg total ms: {payload['total_ms']['avg']:.3f}\n"
        f"- p50 total ms: {payload['total_ms']['p50']:.3f}\n"
        f"- p95 total ms: {payload['total_ms']['p95']:.3f}\n"
        f"- p99 total ms: {payload['total_ms']['p99']:.3f}\n\n"
        "## Breakdown\n\n"
        f"- provider forward avg ms: {payload['provider_forward_ms']['avg']:.3f}\n"
        f"- feature extraction avg ms: {payload['feature_extraction_ms']['avg']:.3f}\n"
        f"- specialist scoring avg ms: {payload['specialist_scoring_ms']['avg']:.3f}\n"
        f"- blend and formatting avg ms: {payload['blend_and_formatting_ms']['avg']:.3f}\n"
    )


def _select_historical_specialist_feature_subset(
    *,
    specialist_rows: list[dict[str, float]],
    prefix: str,
) -> list[dict[str, float]]:
    return [
        _select_historical_specialist_feature_row(
            specialist_row=row,
            prefix=prefix,
        )
        for row in specialist_rows
    ]


def _select_historical_specialist_feature_row(
    *,
    specialist_row: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    return {
        name: value
        for name, value in specialist_row.items()
        if name.startswith(prefix)
        or name in {
            "specialist_internal_disagreement_hint",
            "specialist_internal_consistency_hint",
        }
    }


def _extract_historical_specialist_features(
    *,
    prompt: str,
    response: str,
    token_stats: list[TokenUncertaintyStat] | None,
    internal_signal: InternalModelSignal | None,
) -> dict[str, float]:
    stats = token_stats or []
    tokens = [stat.token for stat in stats]
    token_count = max(len(tokens), 1)
    numeric_indices = [index for index, token in enumerate(tokens) if any(character.isdigit() for character in token)]
    entity_indices = [
        index
        for index, token in enumerate(tokens)
        if _historical_titlecase_token(token)
    ]
    logprobs = [float(stat.logprob) for stat in stats]
    entropies = [float(stat.entropy) for stat in stats]
    margins = [float(stat.top1_top2_margin) for stat in stats]
    return {
        "specialist_numeric_density": len(numeric_indices) / token_count,
        "specialist_numeric_span_count": float(len(numeric_indices)),
        "specialist_numeric_margin_mean": _subset_mean(margins, numeric_indices),
        "specialist_numeric_margin_min": _subset_min(margins, numeric_indices),
        "specialist_numeric_entropy_mean": _subset_mean(entropies, numeric_indices),
        "specialist_numeric_logprob_asymmetry": _subset_mean(logprobs, numeric_indices) - _mean(logprobs),
        "specialist_numeric_tail_suspicion": _tail_suspicion(
            indices=numeric_indices,
            margins=margins,
            entropies=entropies,
        ),
        "specialist_entity_density": len(entity_indices) / token_count,
        "specialist_entity_margin_mean": _subset_mean(margins, entity_indices),
        "specialist_entity_margin_min": _subset_min(margins, entity_indices),
        "specialist_entity_entropy_mean": _subset_mean(entropies, entity_indices),
        "specialist_entity_confidence_dip": _mean(margins) - _subset_mean(margins, entity_indices),
        "specialist_entity_segment_suspicion": _segment_suspicion(entity_indices, entropies, margins),
        "specialist_long_length_bucket": _length_bucket(response),
        "specialist_long_entropy_drift": _tail_mean(entropies) - _head_mean(entropies),
        "specialist_long_margin_drift": _head_mean(margins) - _tail_mean(margins),
        "specialist_long_segment_variance": _segment_variance(entropies),
        "specialist_long_max_suspicious_segment_score": _max_suspicious_segment_score(entropies, margins),
        "specialist_long_longest_suspicious_run": float(_longest_suspicious_run(entropies, margins)),
        "specialist_internal_disagreement_hint": float(internal_signal.layer_disagreement_mean) if internal_signal else 0.0,
        "specialist_internal_consistency_hint": float(internal_signal.early_late_layer_consistency) if internal_signal else 0.0,
        "specialist_bucket_hint_numbers": 1.0 if ("how many" in prompt.lower() or any(character.isdigit() for character in response)) else 0.0,
        "specialist_bucket_hint_entities": 1.0 if entity_indices else 0.0,
        "specialist_bucket_hint_long": 1.0 if len(response.split()) > 18 else 0.0,
    }


def _collect_signals(*, token_stat_provider, prompt: str, response: str):
    if hasattr(token_stat_provider, "collect_signals"):
        collected = token_stat_provider.collect_signals(prompt=prompt, response=response)
        return collected.token_stats, collected.internal_signal
    return token_stat_provider.collect(prompt=prompt, response=response), None


def _historical_titlecase_token(token: str) -> bool:
    normalized = token.strip("Ġ▁ ,.;:!?()[]{}\"'")
    return bool(normalized[:1].isupper() and any(character.isalpha() for character in normalized))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _subset_mean(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return _mean(selected)


def _subset_min(values: list[float], indices: list[int]) -> float:
    selected = [values[index] for index in indices if index < len(values)]
    return min(selected) if selected else 0.0


def _tail_suspicion(*, indices: list[int], margins: list[float], entropies: list[float]) -> float:
    if not indices:
        return 0.0
    split = max(1, len(margins) // 2)
    tail_indices = [index for index in indices if index >= split]
    if not tail_indices:
        return 0.0
    return _subset_mean(entropies, tail_indices) - _subset_mean(margins, tail_indices)


def _segment_suspicion(indices: list[int], entropies: list[float], margins: list[float]) -> float:
    if not indices:
        return 0.0
    segment_size = max(1, len(entropies) // 3)
    segment_scores: list[float] = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_indices = [index for index in indices if start <= index < end]
        if not segment_indices:
            continue
        segment_scores.append(
            _subset_mean(entropies, segment_indices) - _subset_mean(margins, segment_indices)
        )
    return max(segment_scores) if segment_scores else 0.0


def _length_bucket(response: str) -> float:
    token_count = len(response.split())
    if token_count <= 8:
        return 0.0
    if token_count <= 20:
        return 1.0
    return 2.0


def _tail_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _mean(values[-max(1, len(values) // 3) :])


def _head_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _mean(values[: max(1, len(values) // 3)])


def _segment_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    segment_size = max(1, len(values) // 3)
    means = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(values) if segment_index == 2 else min(len(values), start + segment_size)
        segment = values[start:end]
        if segment:
            means.append(_mean(segment))
    overall = _mean(means)
    return _mean([(value - overall) ** 2 for value in means])


def _max_suspicious_segment_score(entropies: list[float], margins: list[float]) -> float:
    if not entropies or not margins:
        return 0.0
    segment_size = max(1, len(entropies) // 3)
    scores = []
    for segment_index in range(3):
        start = segment_index * segment_size
        end = len(entropies) if segment_index == 2 else min(len(entropies), start + segment_size)
        segment_entropies = entropies[start:end]
        segment_margins = margins[start:end]
        if segment_entropies and segment_margins:
            scores.append(_mean(segment_entropies) - _mean(segment_margins))
    return max(scores) if scores else 0.0


def _longest_suspicious_run(entropies: list[float], margins: list[float]) -> int:
    longest = 0
    current = 0
    for entropy, margin in zip(entropies, margins):
        if entropy >= 0.7 or margin <= 0.15:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
