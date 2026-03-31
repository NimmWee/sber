from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.default_detector import build_default_detector_extractor
from eval.error_analysis import DefaultDetectorErrorAnalysisRunner
from eval.runner import RawExampleEvaluationDataset
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    build_ablation_examples,
    resolve_transformers_provider_config,
    write_json_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "error_analysis_real_provider"),
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_examples, validation_examples = build_ablation_examples()
    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    runner = DefaultDetectorErrorAnalysisRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=build_default_detector_extractor(),
            token_stat_provider=provider,
        ),
        artifact_dir=artifact_dir / "default_detector",
    )
    summary = runner.run()

    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="error_analysis_real_provider_summary.json",
        payload={
            "model_source": config.model_source,
            "response_delimiter": config.response_delimiter,
            "pr_auc": summary.pr_auc,
            "sample_size": summary.sample_size,
            "false_positive_count": summary.false_positive_count,
            "false_negative_count": summary.false_negative_count,
            "non_trivial_buckets": summary.non_trivial_buckets,
            "hardest_examples": [
                {
                    "prompt": example.prompt,
                    "response": example.response,
                    "label": example.label,
                    "predicted_label": example.predicted_label,
                    "probability": example.probability,
                    "mistake_confidence": example.mistake_confidence,
                    "buckets": example.buckets,
                }
                for example in summary.hardest_examples
            ],
            "model_artifact_path": summary.model_artifact_path,
            "summary_artifact_path": summary.summary_artifact_path,
        },
    )

    print(f"model={config.model_source}")
    print(f"pr_auc={summary.pr_auc:.4f}")
    print(f"false_positives={summary.false_positive_count}")
    print(f"false_negatives={summary.false_negative_count}")
    print(f"non_trivial_buckets={','.join(summary.non_trivial_buckets)}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
