import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.error_analysis import DefaultDetectorErrorAnalysisRunner
from eval.runner import RawExampleEvaluationDataset, RawLabeledExample
from features.extractor import InternalModelSignal, TokenUncertaintyStat
from eval.default_detector import build_default_detector_extractor


def test_default_detector_error_analysis_runner_writes_summary_for_new_default_detector(
    tmp_path,
) -> None:
    dataset = RawExampleEvaluationDataset(
        train_examples=[
            RawLabeledExample(
                prompt="Who wrote Hamlet?",
                response="William Shakespeare wrote Hamlet.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("William", -0.05, 0.12, 0.75),
                    TokenUncertaintyStat("Shakespeare", -0.09, 0.16, 0.63),
                ],
                internal_signal=InternalModelSignal(1.0, 0.3, 0.05, 0.08, 0.12, 0.85),
            ),
            RawLabeledExample(
                prompt="What is the capital of Italy?",
                response="The capital of Italy is Verona.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("Verona", -1.20, 0.91, 0.08),
                    TokenUncertaintyStat("Italy", -0.20, 0.24, 0.50),
                ],
                internal_signal=InternalModelSignal(1.3, 0.4, 0.10, 0.18, 0.24, 0.70),
            ),
            RawLabeledExample(
                prompt="What is the capital of France?",
                response="Paris is the capital of France.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Paris", -0.04, 0.10, 0.79),
                    TokenUncertaintyStat("France", -0.08, 0.14, 0.68),
                ],
                internal_signal=InternalModelSignal(1.0, 0.3, 0.05, 0.07, 0.10, 0.87),
            ),
            RawLabeledExample(
                prompt="When was the treaty signed?",
                response="The treaty was signed on 1492-13-40.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("1492-13-40", -1.35, 0.98, 0.06),
                    TokenUncertaintyStat("signed", -0.22, 0.25, 0.47),
                ],
                internal_signal=InternalModelSignal(1.4, 0.45, 0.12, 0.22, 0.30, 0.66),
            ),
        ],
        validation_examples=[
            RawLabeledExample(
                prompt="What is the capital of Germany?",
                response="Munich is the capital of Germany.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("Munich", -1.10, 0.88, 0.09),
                    TokenUncertaintyStat("Germany", -0.12, 0.17, 0.62),
                ],
                internal_signal=InternalModelSignal(1.35, 0.42, 0.11, 0.19, 0.26, 0.72),
            ),
            RawLabeledExample(
                prompt="Who discovered penicillin?",
                response="Alexander Fleming discovered penicillin.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Alexander", -0.05, 0.11, 0.77),
                    TokenUncertaintyStat("Fleming", -0.08, 0.13, 0.69),
                ],
                internal_signal=InternalModelSignal(0.95, 0.28, 0.04, 0.06, 0.09, 0.90),
            ),
        ],
        extractor=build_default_detector_extractor(),
    )

    summary = DefaultDetectorErrorAnalysisRunner(
        dataset=dataset,
        artifact_dir=tmp_path,
    ).run()

    assert 0.0 <= summary.pr_auc <= 1.0
    assert summary.sample_size == 2
    assert summary.model_artifact_path is not None
    assert summary.summary_artifact_path is not None
    assert Path(summary.summary_artifact_path).exists()

    model_payload = json.loads(Path(summary.model_artifact_path).read_text())
    assert "token_mean_logprob" in model_payload["feature_names"]
    assert "token_logprob_std" not in model_payload["feature_names"]
    assert "internal_last_layer_pooled_l2" in model_payload["feature_names"]
    assert "internal_selected_layer_disagreement_max" in model_payload["feature_names"]

    summary_payload = json.loads(Path(summary.summary_artifact_path).read_text())
    assert "false_positive_count" in summary_payload
    assert "false_negative_count" in summary_payload
    assert "hardest_examples" in summary_payload
    assert "bucket_summaries" in summary_payload
    assert "focused_bucket_summaries" in summary_payload
    assert "recommended_next_improvement" in summary_payload
