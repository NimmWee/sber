from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark_ablation import run_public_benchmark_ablation
from eval.runner import RawLabeledExample
from features.extractor import TokenUncertaintyStat


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


def test_run_public_benchmark_ablation_writes_summary(tmp_path) -> None:
    csv_path = tmp_path / "knowledge_bench_public.csv"
    csv_path.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\n"
        "Q1,Correct answer,False,Ref,\n"
        "Q2,Wrong answer,True,Ref,\n"
        "Q3,Water boils at 100 C.,False,Ref,\n"
        "Q4,Water boils at 250 C.,True,Ref,\n",
        encoding="utf-8",
    )

    summary = run_public_benchmark_ablation(
        dataset_path=csv_path,
        train_examples=[
            RawLabeledExample(prompt="Q1", response="Correct answer", label=0),
            RawLabeledExample(prompt="Q2", response="Wrong answer", label=1),
            RawLabeledExample(prompt="Q3", response="Water boils at 100 C.", label=0),
            RawLabeledExample(prompt="Q4", response="Water boils at 250 C.", label=1),
        ],
        token_stat_provider=FakeTokenStatProvider(),
        artifact_dir=tmp_path / "artifacts",
    )

    assert "base_token_uncertainty" in summary["variants"]
    assert "extended_token_uncertainty" in summary["variants"]
    assert "internal_features" in summary["variants"]
    assert "internal_features_lightgbm" in summary["variants"]
    assert "best_variant" in summary
    assert Path(summary["artifact_path"]).exists()
