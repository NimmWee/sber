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
from utils.script_helpers import write_json_artifact


HISTORICAL_BEST_COMMIT = "d3fa946"
HISTORICAL_BEST_VARIANT = "baseline_plus_all_specialists"
HISTORICAL_BEST_PR_AUC = 0.6881
FROZEN_BLEND_WEIGHTS = {
    "baseline": 0.55,
    "numeric": 0.15,
    "entity": 0.15,
    "long": 0.15,
}


@dataclass(frozen=True)
class FrozenSubmissionBundle:
    baseline_head: Any
    numeric_head: Any
    entity_head: Any
    long_head: Any
    metadata: dict[str, Any]


def build_frozen_best_metadata() -> dict[str, Any]:
    return {
        "historical_best_commit": HISTORICAL_BEST_COMMIT,
        "historical_best_variant": HISTORICAL_BEST_VARIANT,
        "historical_best_pr_auc": HISTORICAL_BEST_PR_AUC,
        "blend_weights": dict(FROZEN_BLEND_WEIGHTS),
    }


def train_frozen_best_submission(
    *,
    dataset_path: str | Path,
    token_stat_provider,
    artifact_dir: str | Path,
) -> dict[str, Any]:
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
        **build_frozen_best_metadata(),
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
    label_threshold: float = 0.3,
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
        raise ValueError("output_mode must be 'probability' or 'boolean'")

    extractor = _build_frozen_best_extractor()
    output_csv_path = Path(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
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
        if output_mode == "boolean":
            scored_rows.append(
                {
                    **row,
                    "hallucination": "true" if probability >= float(label_threshold) else "false",
                }
            )
        else:
            scored_rows.append({**row, "hallucination_probability": probability})

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)
    return {
        "sample_size": len(scored_rows),
        "output_path": str(output_csv_path),
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
