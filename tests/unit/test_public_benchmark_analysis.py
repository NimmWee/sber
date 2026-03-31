from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark_analysis import analyze_public_benchmark_scores


def test_analyze_public_benchmark_scores_reports_overlap_and_bias(tmp_path) -> None:
    score_path = tmp_path / "knowledge_bench_public_scores.json"
    score_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"label": 0, "probability": 0.10},
                    {"label": 0, "probability": 0.20},
                    {"label": 1, "probability": 0.15},
                    {"label": 1, "probability": 0.25},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_public_benchmark_scores(score_path)

    assert summary["sample_size"] == 4
    assert summary["negative_count"] == 2
    assert summary["positive_count"] == 2
    assert summary["score_overlap"] > 0.0
    assert summary["is_biased_low"] is True
