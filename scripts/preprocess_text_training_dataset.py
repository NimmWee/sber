from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_preprocessing import preprocess_textual_training_dataset
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
        default=str(PROJECT_ROOT / "artifacts" / "textual_training_preprocessed"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = preprocess_textual_training_dataset(
        dataset_path=args.dataset_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )

    print(f"model={config.model_source}")
    print(f"sample_size={summary['summary']['sample_size']}")
    print(f"feature_count={len(summary['summary']['feature_names'])}")
    print(f"artifact={summary['artifact_path']}")


if __name__ == "__main__":
    main()
