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
from utils.script_helpers import (
    resolve_frozen_submission_config,
    resolve_transformers_provider_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score ready-made prompt/response pairs with the frozen hallucination detector."
    )
    parser.add_argument("--config", default=None, help="Path to provider config JSON.")
    parser.add_argument(
        "--submission-config",
        default=None,
        help="Path to frozen submission metadata config JSON.",
    )
    parser.add_argument(
        "--input-path",
        default=str(PROJECT_ROOT / "data" / "bench" / "knowledge_bench_private.csv"),
        help="CSV with prompt and response columns.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
        help="Directory with trained frozen artifacts.",
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "bench" / "knowledge_bench_private_scores.csv"),
        help="Destination CSV path for scores or boolean labels.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("probability", "boolean"),
        default="probability",
        help="Probability is the primary competition mode; boolean is an optional serving mode.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help="Serving threshold for boolean mode only. Does not affect PR-AUC.",
    )
    args = parser.parse_args()

    try:
        submission_config = resolve_frozen_submission_config(
            project_root=PROJECT_ROOT,
            explicit_config_path=args.submission_config,
        )
        label_threshold = (
            float(args.label_threshold)
            if args.label_threshold is not None
            else float(submission_config.get("serving_threshold", 0.3))
        )
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
            output_mode=args.output_mode,
            label_threshold=label_threshold,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Scoring setup error: {error}") from error
    except RuntimeError as error:
        raise SystemExit(
            f"Scoring runtime error: {error}. Check that the local checkpoint is available and the config path is correct."
        ) from error

    print(f"model={config.model_source}")
    print(f"sample_size={summary['sample_size']}")
    print(f"output={summary['output_path']}")
    print(f"output_mode={args.output_mode}")
    print(f"blend_version={summary['metadata'].get('blend_version')}")
    print(f"blend_weight_source={summary['metadata'].get('blend_weight_source')}")
    print(f"serving_threshold={label_threshold}")


if __name__ == "__main__":
    main()
