from __future__ import annotations

import json
from pathlib import Path


def analyze_public_benchmark_scores(score_path: str | Path) -> dict[str, float | int | bool]:
    payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("knowledge_bench_public_scores.json must contain non-empty rows")

    labels: list[int] = []
    probabilities: list[float] = []
    for row in rows:
        label = int(row["label"])
        probability = float(row["probability"])
        labels.append(label)
        probabilities.append(probability)

    positive_scores = [probability for probability, label in zip(probabilities, labels) if label == 1]
    negative_scores = [probability for probability, label in zip(probabilities, labels) if label == 0]
    if not positive_scores or not negative_scores:
        raise ValueError("score rows must contain both positive and negative examples")

    positive_mean = _mean(positive_scores)
    negative_mean = _mean(negative_scores)
    score_overlap = max(0.0, min(max(negative_scores), max(positive_scores)) - max(min(negative_scores), min(positive_scores)))
    is_biased_low = positive_mean < 0.5 and negative_mean < 0.5

    return {
        "sample_size": len(rows),
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "positive_mean_score": positive_mean,
        "negative_mean_score": negative_mean,
        "positive_max_score": max(positive_scores),
        "negative_max_score": max(negative_scores),
        "positive_min_score": min(positive_scores),
        "negative_min_score": min(negative_scores),
        "score_overlap": score_overlap,
        "is_biased_low": is_biased_low,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
