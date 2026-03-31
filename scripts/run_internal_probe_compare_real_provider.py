from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.internal_compare import compare_base_vs_internal_features
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    build_ablation_examples,
    resolve_transformers_provider_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "internal_probe_compare_real_provider"),
    )
    parser.add_argument("--latency-repeat-count", type=int, default=20)
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    base_provider = TransformersTokenStatProvider(
        config=replace(config, enable_internal_features=False),
    )
    internal_provider = TransformersTokenStatProvider(
        config=replace(
            config,
            enable_internal_features=True,
            selected_hidden_layers=(-1, -2),
        ),
    )
    train_examples, validation_examples = build_ablation_examples()
    summary = compare_base_vs_internal_features(
        train_examples=train_examples,
        validation_examples=validation_examples,
        base_provider=base_provider,
        internal_provider=internal_provider,
        artifact_dir=args.artifact_dir,
        latency_repeat_count=args.latency_repeat_count,
    )

    print(f"model={config.model_source}")
    print(f"baseline_pr_auc={summary['baseline']['pr_auc']:.4f}")
    print(f"internal_pr_auc={summary['internal']['pr_auc']:.4f}")
    print(f"pr_auc_delta={summary['pr_auc_delta']:.4f}")
    print(f"latency_delta_ms={summary['latency_delta_ms']:.4f}")
    print(f"recommendation={summary['recommendation']}")
    print(f"artifact={summary['artifact_path']}")


if __name__ == "__main__":
    main()
