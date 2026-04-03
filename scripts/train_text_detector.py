from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_preprocessing import (
    preprocess_textual_training_dataset,
    train_detector_from_preprocessed_rows,
)
from inference.token_stats import TransformersTokenStatProvider
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
        default=str(PROJECT_ROOT / "artifacts" / "text_training_preprocessed"),
    )
    parser.add_argument(
        "--model-output-path",
        default=str(PROJECT_ROOT / "model" / "default_detector_lightgbm.json"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    preprocessed = preprocess_textual_training_dataset(
        dataset_path=args.dataset_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )
    training_summary = train_detector_from_preprocessed_rows(
        preprocessed_rows=preprocessed["rows"],
        model_output_path=args.model_output_path,
    )

    print(f"model={config.model_source}")
    print(f"train_size={training_summary['train_size']}")
    print(f"dev_size={training_summary['dev_size']}")
    print(f"dev_pr_auc={training_summary['dev_pr_auc']:.4f}")
    print(f"features_artifact={preprocessed['artifact_path']}")
    print(f"model_artifact={training_summary['model_artifact_path']}")


if __name__ == "__main__":
    main()
