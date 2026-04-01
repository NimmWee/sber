from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.non_public_recovery import run_non_public_retraining_public_eval
from eval.runner import RawLabeledExample
from features.extractor import InternalModelSignal, TokenUncertaintyStat
from eval.default_detector import train_default_detector_head


class FakeSignalProvider:
    def collect(self, prompt: str, response: str):
        return self.collect_signals(prompt=prompt, response=response).token_stats

    def collect_signals(self, prompt: str, response: str):
        from inference.token_stats import CollectedModelSignals

        response_lower = response.lower()
        is_hallucinated = any(
            marker in response_lower
            for marker in ["munich", "napoleon", "250", "2031", "microsoft", "11 planets"]
        )
        if is_hallucinated:
            token_stats = [
                TokenUncertaintyStat("wrong", -1.1, 0.9, 0.1),
                TokenUncertaintyStat("fact", -0.9, 0.7, 0.2),
            ]
            internal_signal = InternalModelSignal(1.4, 0.45, 0.12, 0.22, 0.30, 0.66)
        else:
            token_stats = [
                TokenUncertaintyStat("correct", -0.1, 0.2, 0.7),
                TokenUncertaintyStat("fact", -0.15, 0.25, 0.6),
            ]
            internal_signal = InternalModelSignal(1.0, 0.30, 0.05, 0.08, 0.12, 0.88)
        return CollectedModelSignals(token_stats=token_stats, internal_signal=internal_signal)


def test_run_non_public_retraining_public_eval_writes_before_after_summary(tmp_path) -> None:
    public_csv = tmp_path / "knowledge_bench_public.csv"
    public_csv.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\n"
        "What is the capital of Germany?,Berlin is the capital of Germany.,False,Berlin,\n"
        "What is the capital of Germany?,Munich is the capital of Germany.,True,Berlin,\n"
        "What is the boiling point of water?,Water boils at 100 degrees Celsius.,False,100,\n"
        "What is the boiling point of water?,Water boils at 250 degrees Celsius.,True,100,\n",
        encoding="utf-8",
    )

    baseline_head = train_default_detector_head(
        feature_rows=[
            {
                "response_length": 20.0,
                "token_mean_logprob": -0.1,
                "token_min_logprob": -0.15,
                "token_entropy_mean": 0.2,
                "token_top1_top2_margin_mean": 0.7,
                "token_tail_low_confidence_rate": 0.0,
                "token_confidence_decay": 0.02,
                "internal_last_layer_pooled_l2": 1.0,
                "internal_last_layer_pooled_mean_abs": 0.3,
                "internal_selected_layer_norm_variance": 0.05,
                "internal_layer_disagreement_mean": 0.08,
                "internal_selected_layer_disagreement_max": 0.12,
                "internal_early_late_layer_consistency": 0.88,
                "internal_entropy_disagreement_gap": 0.12,
                "internal_low_confidence_disagreement_gap": 0.08,
            },
            {
                "response_length": 24.0,
                "token_mean_logprob": -1.1,
                "token_min_logprob": -1.3,
                "token_entropy_mean": 0.9,
                "token_top1_top2_margin_mean": 0.1,
                "token_tail_low_confidence_rate": 1.0,
                "token_confidence_decay": -0.10,
                "internal_last_layer_pooled_l2": 1.4,
                "internal_last_layer_pooled_mean_abs": 0.45,
                "internal_selected_layer_norm_variance": 0.12,
                "internal_layer_disagreement_mean": 0.22,
                "internal_selected_layer_disagreement_max": 0.30,
                "internal_early_late_layer_consistency": 0.66,
                "internal_entropy_disagreement_gap": 0.68,
                "internal_low_confidence_disagreement_gap": 0.78,
            },
        ],
        labels=[0, 1],
    )
    baseline_artifact_path = tmp_path / "baseline_lightgbm_head.json"
    baseline_head.save(baseline_artifact_path)

    summary = run_non_public_retraining_public_eval(
        public_dataset_path=public_csv,
        baseline_model_artifact_path=baseline_artifact_path,
        token_stat_provider=FakeSignalProvider(),
        artifact_dir=tmp_path / "artifacts",
    )

    assert (
        summary["dataset_summary"]["non_hallucination_count"]
        > summary["dataset_summary"]["hallucination_count"]
    )
    assert "before" in summary["public_benchmark"]
    assert "after" in summary["public_benchmark"]
    assert "pr_auc" in summary["public_benchmark"]["before"]
    assert "pr_auc" in summary["public_benchmark"]["after"]
    assert "numbers" in summary["public_benchmark"]["bucket_deltas"]
    assert "entity_like_tokens" in summary["public_benchmark"]["bucket_deltas"]
    assert "places" in summary["public_benchmark"]["bucket_deltas"]
    assert "long_responses" in summary["public_benchmark"]["bucket_deltas"]
    assert summary["recall_recovery"]["false_negatives_decreased"] in {True, False}
    assert "false_positive_increase_too_much" in summary["recall_recovery"]
    assert "accept_change" in summary["decision"]
    assert "rejection_reason" in summary["decision"]
    assert Path(summary["artifact_path"]).exists()
    assert Path(summary["trained_model_artifact_path"]).exists()

    payload = json.loads(Path(summary["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["dataset_summary"]["corruption_taxonomy"]["number_nearby"] > 0
    assert "recall_recovery" in payload
    assert "decision" in payload
