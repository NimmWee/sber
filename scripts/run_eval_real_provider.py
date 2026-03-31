from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.default_detector import build_default_detector_extractor
from eval.runner import RawExampleEvaluationDataset, TrainValidationEvaluationRunner
from inference.token_stats import TransformersTokenStatProvider
from models.head import TrainedLogisticRegressionHead
from utils.script_helpers import (
    build_smoke_examples,
    resolve_transformers_provider_config,
    write_json_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "eval_real_provider"),
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_examples, validation_examples = build_smoke_examples()
    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )

    baseline_summary = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=build_default_detector_extractor(),
        ),
        artifact_dir=artifact_dir / "baseline",
    ).run()

    provider = TransformersTokenStatProvider(config=config)
    provider_summary = TrainValidationEvaluationRunner(
        dataset=RawExampleEvaluationDataset(
            train_examples=train_examples,
            validation_examples=validation_examples,
            extractor=build_default_detector_extractor(),
            token_stat_provider=provider,
        ),
        artifact_dir=artifact_dir / "provider",
    ).run()

    provider_model_payload = json.loads(Path(provider_summary.model_artifact_path).read_text())

    payload = {
        "model_source": config.model_source,
        "response_delimiter": config.response_delimiter,
        "baseline_pr_auc": baseline_summary.pr_auc,
        "provider_pr_auc": provider_summary.pr_auc,
        "pr_auc_delta": provider_summary.pr_auc - baseline_summary.pr_auc,
        "default_detector_feature_names": provider_model_payload["feature_names"],
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
