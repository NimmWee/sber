from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark_ablation import run_public_benchmark_ablation
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    build_ablation_examples,
    resolve_public_benchmark_path,
    resolve_transformers_provider_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "public_benchmark_ablation"),
    )
    parser.add_argument("--latency-repeat-count", type=int, default=5)
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
    train_examples, _ = build_ablation_examples()

    summary = run_public_benchmark_ablation(
        dataset_path=dataset_path,
        train_examples=train_examples,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
        latency_repeat_count=args.latency_repeat_count,
    )

    print(f"dataset={dataset_path}")
    print(f"model={config.model_source}")
    print(f"sample_size={summary['sample_size']}")
    print(f"best_variant={summary['best_variant']}")
    print(
        "signal_collection_runtime_ms="
        f"{summary.get('signal_collection_runtime_ms', 0.0):.4f}"
    )
    print(
        "estimated_signal_runtime_improvement_ms="
        f"{summary.get('estimated_signal_runtime_improvement_ms', 0.0):.4f}"
    )
    for variant_name, variant in summary["variants"].items():
        print(
            f"{variant_name} "
            f"pr_auc={variant['pr_auc']:.4f} "
            f"precision={variant.get('precision', 0.0):.4f} "
            f"recall={variant.get('recall', 0.0):.4f} "
            f"predicted_positive_rate={variant.get('predicted_positive_rate', 0.0):.4f} "
            f"latency_mean_ms={variant['latency_total_mean_ms']:.4f} "
            f"false_positives={variant['false_positive_count']} "
            f"false_negatives={variant['false_negative_count']} "
            f"error_buckets={','.join(variant['non_trivial_buckets'])}"
        )
    print(f"artifact={summary['artifact_path']}")


if __name__ == "__main__":
    main()
