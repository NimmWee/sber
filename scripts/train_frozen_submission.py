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
from utils.script_helpers import (
    resolve_frozen_submission_config,
    resolve_transformers_provider_config,
    set_global_random_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen best hallucination detector from text-based train/internal-val data."
    )
    parser.add_argument("--config", default=None, help="Path to provider config JSON.")
    parser.add_argument(
        "--submission-config",
        default=None,
        help="Path to frozen submission metadata config JSON.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data" / "processed" / "textual_training_dataset.jsonl"),
        help="Path to the text-based processed training dataset.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
        help="Directory where trained frozen artifacts will be written.",
    )
    args = parser.parse_args()

    try:
        submission_config = resolve_frozen_submission_config(
            project_root=PROJECT_ROOT,
            explicit_config_path=args.submission_config,
        )
        set_global_random_seed(int(submission_config.get("random_seed", 0)))
        config = resolve_transformers_provider_config(
            project_root=PROJECT_ROOT,
            explicit_config_path=args.config,
        )
        provider = TransformersTokenStatProvider(config=config)
        summary = train_frozen_best_submission(
            dataset_path=args.dataset_path,
            token_stat_provider=provider,
            artifact_dir=args.artifact_dir,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Training setup error: {error}") from error
    except RuntimeError as error:
        raise SystemExit(
            f"Training runtime error: {error}. Check that the local checkpoint is available and the config path is correct."
        ) from error

    print(f"model={config.model_source}")
    print(f"submission_variant={submission_config['historical_best_variant']}")
    print(f"train_size={summary['train_size']}")
    print(f"dev_size={summary['dev_size']}")
    print(f"dev_pr_auc={summary['dev_pr_auc']:.4f}")
    print(f"artifact_dir={summary['artifact_dir']}")
    print(f"metadata_path={summary['metadata_path']}")


if __name__ == "__main__":
    main()
