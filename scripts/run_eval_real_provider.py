from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.runner import RawExampleEvaluationDataset, TrainValidationEvaluationRunner
from features.extractor import StructuralFeatureExtractor
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    build_smoke_examples,
    load_transformers_provider_config,
    write_json_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "token_stat_provider.local.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "eval_real_provider"),
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_examples, validation_examples = build_smoke_examples()
    config = load_transformers_provider_config(args.config)

    baseline_summary = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=StructuralFeatureExtractor(),
        ),
        artifact_dir=artifact_dir / "baseline",
    ).run()

    provider = TransformersTokenStatProvider(config=config)
    provider_summary = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=StructuralFeatureExtractor(enable_token_uncertainty=True),
            token_stat_provider=provider,
        ),
        artifact_dir=artifact_dir / "provider",
    ).run()

    payload = {
        "model_source": config.model_source,
        "response_delimiter": config.response_delimiter,
        "baseline_pr_auc": baseline_summary.pr_auc,
        "provider_pr_auc": provider_summary.pr_auc,
        "pr_auc_delta": provider_summary.pr_auc - baseline_summary.pr_auc,
        "baseline_summary_artifact_path": baseline_summary.summary_artifact_path,
        "provider_summary_artifact_path": provider_summary.summary_artifact_path,
    }
    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="eval_real_provider_summary.json",
        payload=payload,
    )

    print(f"baseline_pr_auc={baseline_summary.pr_auc:.4f}")
    print(f"provider_pr_auc={provider_summary.pr_auc:.4f}")
    print(f"pr_auc_delta={provider_summary.pr_auc - baseline_summary.pr_auc:.4f}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
