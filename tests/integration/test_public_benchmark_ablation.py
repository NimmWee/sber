from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark_ablation import run_public_benchmark_ablation
from eval.runner import RawLabeledExample
from features.extractor import TokenUncertaintyStat
from inference.token_stats import CollectedModelSignals
from features.extractor import InternalModelSignal


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


class CountingSignalProvider:
    def __init__(self) -> None:
        self.collect_calls = 0
        self.collect_signals_calls = 0

    def collect(self, prompt: str, response: str):
        self.collect_calls += 1
        return self.collect_signals(prompt=prompt, response=response).token_stats

    def collect_signals(self, prompt: str, response: str):
        self.collect_signals_calls += 1
        return CollectedModelSignals(
            token_stats=[
                TokenUncertaintyStat("fact", -0.1, 0.2, 0.7),
                TokenUncertaintyStat("wrong", -1.0, 0.8, 0.1),
            ],
            internal_signal=InternalModelSignal(
                last_layer_pooled_l2=1.2,
                last_layer_pooled_mean_abs=0.4,
                selected_layer_norm_variance=0.08,
                layer_disagreement_mean=0.12,
                selected_layer_disagreement_max=0.16,
                early_late_layer_consistency=0.82,
            ),
        )


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
    assert "baseline_plus_numeric_specialist" in summary["variants"]
    assert "baseline_plus_entity_specialist" in summary["variants"]
    assert "baseline_plus_long_specialist" in summary["variants"]
    assert "baseline_plus_consistency_specialist" in summary["variants"]
    assert "baseline_plus_stability_specialist" in summary["variants"]
    assert "baseline_plus_all_specialists" in summary["variants"]
    assert "weighted_independent_score_ensemble" in summary["variants"]
    assert "weighted_independent_score_ensemble_b70_n20_l10" in summary["variants"]
    assert "weighted_independent_score_ensemble_b65_n25_l10" in summary["variants"]
    assert "weighted_independent_score_ensemble_b65_n20_l15" in summary["variants"]
    assert "weighted_independent_score_ensemble_b60_n25_l15" in summary["variants"]
    assert "fused_specialist_ensemble" in summary["variants"]
    assert "precision" in summary["variants"]["fused_specialist_ensemble"]
    assert "recall" in summary["variants"]["fused_specialist_ensemble"]
    assert "predicted_positive_rate" in summary["variants"]["fused_specialist_ensemble"]
    assert "score_distribution" in summary["variants"]["fused_specialist_ensemble"]
    assert "feature_importance" in summary["variants"]["baseline_plus_numeric_specialist"]
    assert "bucket_summaries" in summary["variants"]["baseline_plus_all_specialists"]
    assert "selected_feature_names" in summary["variants"]["baseline_plus_all_specialists"]
    assert "feature_count_before" in summary["variants"]["baseline_plus_all_specialists"]
    assert "feature_count_after" in summary["variants"]["baseline_plus_all_specialists"]
    assert (
        summary["variants"]["baseline_plus_all_specialists"]["feature_count_after"]
        < summary["variants"]["baseline_plus_all_specialists"]["feature_count_before"]
    )
    assert "consistency_specialist_score" in summary["variants"]["baseline_plus_consistency_specialist"]["score_components_mean"]
    assert "score_label_correlation" in summary["variants"]["baseline_plus_consistency_specialist"]
    assert "ensemble_weights" in summary["variants"]["weighted_independent_score_ensemble"]
    assert "scorer_pr_auc" in summary["variants"]["weighted_independent_score_ensemble"]
    assert "scorer_correlations" in summary["variants"]["weighted_independent_score_ensemble"]
    assert "best_weighted_independent_score_ensemble_variant" in summary
    assert "best_weighted_independent_score_ensemble_weights" in summary
    assert "best_variant" in summary
    assert Path(summary["artifact_path"]).exists()


def test_run_public_benchmark_ablation_reuses_cached_model_signals(tmp_path) -> None:
    csv_path = tmp_path / "knowledge_bench_public.csv"
    csv_path.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\n"
        "Q1,Correct answer,False,Ref,\n"
        "Q2,Wrong answer,True,Ref,\n",
        encoding="utf-8",
    )
    provider = CountingSignalProvider()
    train_examples = [
        RawLabeledExample(prompt="Q1", response="Correct answer", label=0),
        RawLabeledExample(prompt="Q2", response="Wrong answer", label=1),
    ]

    summary = run_public_benchmark_ablation(
        dataset_path=csv_path,
        train_examples=train_examples,
        token_stat_provider=provider,
        artifact_dir=tmp_path / "artifacts",
    )

    assert provider.collect_signals_calls == 8
    assert Path(summary["cached_signal_artifact_path"]).exists()
    assert summary["estimated_signal_runtime_improvement_ms"] >= 0.0
