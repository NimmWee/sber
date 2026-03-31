import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.error_analysis import DefaultDetectorErrorAnalysisRunner
from eval.runner import RawExampleEvaluationDataset, RawLabeledExample
from features.extractor import StructuralFeatureExtractor, TokenUncertaintyStat


def test_default_detector_error_analysis_runner_writes_summary_and_keeps_base_token_features_only(
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
            ),
            RawLabeledExample(
                prompt="What is the capital of Italy?",
                response="The capital of Italy is Verona.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("Verona", -1.20, 0.91, 0.08),
                    TokenUncertaintyStat("Italy", -0.20, 0.24, 0.50),
                ],
            ),
            RawLabeledExample(
                prompt="What is the capital of France?",
                response="Paris is the capital of France.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Paris", -0.04, 0.10, 0.79),
                    TokenUncertaintyStat("France", -0.08, 0.14, 0.68),
                ],
            ),
            RawLabeledExample(
                prompt="When was the treaty signed?",
                response="The treaty was signed on 1492-13-40.",
                label=1,
                token_stats=[
                    TokenUncertaintyStat("1492-13-40", -1.35, 0.98, 0.06),
                    TokenUncertaintyStat("signed", -0.22, 0.25, 0.47),
                ],
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
            ),
            RawLabeledExample(
                prompt="Who discovered penicillin?",
                response="Alexander Fleming discovered penicillin.",
                label=0,
                token_stats=[
                    TokenUncertaintyStat("Alexander", -0.05, 0.11, 0.77),
                    TokenUncertaintyStat("Fleming", -0.08, 0.13, 0.69),
                ],
            ),
        ],
        extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
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

    summary_payload = json.loads(Path(summary.summary_artifact_path).read_text())
    assert "false_positive_count" in summary_payload
    assert "false_negative_count" in summary_payload
    assert "hardest_examples" in summary_payload
    assert "bucket_summaries" in summary_payload
