from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_preprocessing import score_private_dataset
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", required=True)
    parser.add_argument(
        "--model-artifact-path",
        default=str(PROJECT_ROOT / "model" / "default_detector_lightgbm.json"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "artifacts" / "private_scores" / "private_scores.csv"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = score_private_dataset(
        input_path=args.input_path,
        model_artifact_path=args.model_artifact_path,
        token_stat_provider=provider,
        output_path=args.output_path,
    )

    print(f"model={config.model_source}")
    print(f"sample_size={summary['sample_size']}")
    print(f"output={summary['output_path']}")


if __name__ == "__main__":
    main()
