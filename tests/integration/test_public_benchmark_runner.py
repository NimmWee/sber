import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark import evaluate_public_benchmark
from features.extractor import TokenUncertaintyStat
from models.head import train_logistic_regression_head


class FakeTokenStatProvider:
    def collect(self, prompt: str, response: str):
        if "wrong" in response.lower() or "250" in response:
            return [
                TokenUncertaintyStat("wrong", -1.2, 0.9, 0.1),
                TokenUncertaintyStat("fact", -0.8, 0.7, 0.2),
            ]
        return [
            TokenUncertaintyStat("correct", -0.1, 0.2, 0.7),
            TokenUncertaintyStat("fact", -0.15, 0.25, 0.6),
        ]


def test_evaluate_public_benchmark_writes_summary_and_per_example_scores(tmp_path) -> None:
    csv_path = tmp_path / "knowledge_bench_public.csv"
    csv_path.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\n"
        "What is the capital of France?,Paris is the capital of France.,False,Paris,\n"
        "What is the capital of Germany?,Munich is the capital of Germany.,True,Berlin,\n"
        "What is the boiling point of water?,Water boils at 100 C.,False,100,\n"
        "What is the boiling point of water?,Water boils at 250 C.,True,100,\n",
        encoding="utf-8",
    )

    train_rows = [
        {
            "response_length": 30.0,
            "token_count_proxy": 6.0,
            "digit_count": 0.0,
            "punctuation_count": 1.0,
            "sentence_count_proxy": 1.0,
            "prompt_response_overlap": 0.5,
            "novelty_ratio_proxy": 0.5,
            "token_mean_logprob": -0.1,
            "token_min_logprob": -0.15,
            "token_entropy_mean": 0.2,
            "token_top1_top2_margin_mean": 0.65,
            "token_tail_low_confidence_rate": 0.0,
            "token_confidence_decay": 0.01,
        },
        {
            "response_length": 35.0,
            "token_count_proxy": 7.0,
            "digit_count": 0.0,
            "punctuation_count": 1.0,
            "sentence_count_proxy": 1.0,
            "prompt_response_overlap": 0.3,
            "novelty_ratio_proxy": 0.7,
            "token_mean_logprob": -1.0,
            "token_min_logprob": -1.2,
            "token_entropy_mean": 0.85,
            "token_top1_top2_margin_mean": 0.15,
            "token_tail_low_confidence_rate": 1.0,
            "token_confidence_decay": -0.1,
        },
    ]
    train_labels = [0, 1]
    head = train_logistic_regression_head(train_rows, train_labels)
    model_artifact_path = tmp_path / "logistic_head.json"
    head.save(model_artifact_path)

    summary = evaluate_public_benchmark(
        dataset_path=csv_path,
        model_artifact_path=model_artifact_path,
        token_stat_provider=FakeTokenStatProvider(),
        artifact_dir=tmp_path / "artifacts",
    )

    assert summary.sample_size == 4
    assert 0.0 <= summary.pr_auc <= 1.0
    assert summary.false_positive_count >= 0
    assert summary.false_negative_count >= 0
    assert Path(summary.summary_artifact_path).exists()
    assert Path(summary.per_example_artifact_path).exists()

    scores_payload = json.loads(Path(summary.per_example_artifact_path).read_text())
    assert len(scores_payload["rows"]) == 4
    assert "probability" in scores_payload["rows"][0]
