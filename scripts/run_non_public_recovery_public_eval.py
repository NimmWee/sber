from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.non_public_recovery import run_non_public_retraining_public_eval
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    resolve_public_benchmark_path,
    resolve_transformers_provider_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--baseline-model-artifact-path",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "public_benchmark_ablation"
            / "improved_internal_features_lightgbm"
            / "logistic_head.json"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "non_public_recovery"),
    )
    args = parser.parse_args()

    dataset_path = resolve_public_benchmark_path(
        project_root=PROJECT_ROOT,
        explicit_dataset_path=args.dataset_path,
    )
    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = run_non_public_retraining_public_eval(
        public_dataset_path=dataset_path,
        baseline_model_artifact_path=args.baseline_model_artifact_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )

    print(f"dataset={dataset_path}")
    print(f"model={config.model_source}")
    print(f"dataset_size={summary['dataset_summary']['sample_size']}")
    print(f"non_hallucination_count={summary['dataset_summary']['non_hallucination_count']}")
    print(f"hallucination_count={summary['dataset_summary']['hallucination_count']}")
    print(
        "effective_hallucination_ratio="
        f"{summary['dataset_summary']['effective_label_balance']['hallucination_ratio']:.4f}"
    )
    print(f"before_pr_auc={summary['public_benchmark']['before']['pr_auc']:.4f}")
    print(f"after_pr_auc={summary['public_benchmark']['after']['pr_auc']:.4f}")
    print(f"before_precision={summary['public_benchmark']['before']['precision']:.4f}")
    print(f"after_precision={summary['public_benchmark']['after']['precision']:.4f}")
    print(f"before_recall={summary['public_benchmark']['before']['recall']:.4f}")
    print(f"after_recall={summary['public_benchmark']['after']['recall']:.4f}")
    print(
        "before_predicted_positive_rate="
        f"{summary['public_benchmark']['before']['predicted_positive_rate']:.4f}"
    )
    print(
        "after_predicted_positive_rate="
        f"{summary['public_benchmark']['after']['predicted_positive_rate']:.4f}"
    )
    print(
        "before_score_mean_hallucination="
        f"{summary['public_benchmark']['before']['score_distribution']['hallucination']['mean']:.4f}"
    )
    print(
        "after_score_mean_hallucination="
        f"{summary['public_benchmark']['after']['score_distribution']['hallucination']['mean']:.4f}"
    )
    print(f"before_false_positives={summary['public_benchmark']['before']['false_positive_count']}")
    print(f"before_false_negatives={summary['public_benchmark']['before']['false_negative_count']}")
    print(f"after_false_positives={summary['public_benchmark']['after']['false_positive_count']}")
    print(f"after_false_negatives={summary['public_benchmark']['after']['false_negative_count']}")
    print(f"bucket_deltas={summary['public_benchmark']['bucket_deltas']}")
    print(
        "false_negatives_decreased="
        f"{summary['recall_recovery']['false_negatives_decreased']}"
    )
    print(
        "false_positive_increase_too_much="
        f"{summary['recall_recovery']['false_positive_increase_too_much']}"
    )
    print(f"precision_change={summary['precision_change']:.4f}")
    print(f"guardrails={summary['guardrails']}")
    print(
        "recovery_blend_weight="
        f"{summary['training_config']['recovery_blend_weight']:.4f}"
    )
    print(f"accept_change={summary['decision']['accept_change']}")
    print(f"rejection_reason={summary['decision']['rejection_reason']}")
    print(f"artifact={summary['artifact_path']}")


if __name__ == "__main__":
    main()
