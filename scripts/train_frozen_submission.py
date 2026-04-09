from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.token_stats import TransformersTokenStatProvider
from submission.frozen_best import train_frozen_best_submission
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data" / "processed" / "textual_training_dataset.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = train_frozen_best_submission(
        dataset_path=args.dataset_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )

    print(f"model={config.model_source}")
    print(f"train_size={summary['train_size']}")
    print(f"dev_size={summary['dev_size']}")
    print(f"dev_pr_auc={summary['dev_pr_auc']:.4f}")
    print(f"artifact_dir={summary['artifact_dir']}")


if __name__ == "__main__":
    main()
