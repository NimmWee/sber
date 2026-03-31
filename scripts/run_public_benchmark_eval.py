from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark import evaluate_public_benchmark
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
        "--model-artifact-path",
        default=str(PROJECT_ROOT / "artifacts" / "eval_real_provider" / "provider" / "logistic_head.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "public_benchmark_eval"),
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
    summary = evaluate_public_benchmark(
        dataset_path=dataset_path,
        model_artifact_path=args.model_artifact_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )

    print(f"dataset={dataset_path}")
    print(f"model={config.model_source}")
    print(f"sample_size={summary.sample_size}")
    print(f"pr_auc={summary.pr_auc:.4f}")
    print(f"false_positives={summary.false_positive_count}")
    print(f"false_negatives={summary.false_negative_count}")
    print(f"non_trivial_buckets={','.join(summary.non_trivial_buckets)}")
    print(f"summary_artifact={summary.summary_artifact_path}")
    print(f"per_example_artifact={summary.per_example_artifact_path}")


if __name__ == "__main__":
    main()
