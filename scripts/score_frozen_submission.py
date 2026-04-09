from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.token_stats import TransformersTokenStatProvider
from submission.frozen_best import score_private_frozen_submission
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--input-path",
        default=str(PROJECT_ROOT / "data" / "bench" / "knowledge_bench_private.csv"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "bench" / "knowledge_bench_private_scores.csv"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = score_private_frozen_submission(
        input_path=args.input_path,
        output_path=args.output_path,
        token_stat_provider=provider,
        artifact_dir=args.artifact_dir,
    )

    print(f"model={config.model_source}")
    print(f"sample_size={summary['sample_size']}")
    print(f"output={summary['output_path']}")


if __name__ == "__main__":
    main()
