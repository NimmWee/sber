from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.token_stats import TransformersTokenStatProvider
from submission.frozen_best import run_internal_ablation_report
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run internal-validation ablations for the frozen hallucination detector."
    )
    parser.add_argument("--config", default=None, help="Path to provider config JSON.")
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data" / "processed" / "textual_training_dataset.jsonl"),
        help="Text-based processed training dataset with train/dev split.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
        help="Directory with frozen artifacts or where frozen artifacts will be stored.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory for ablation reports.",
    )
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = resolve_transformers_provider_config(
            project_root=PROJECT_ROOT,
            explicit_config_path=args.config,
        )
        provider = TransformersTokenStatProvider(config=config)
        summary = run_internal_ablation_report(
            dataset_path=args.dataset_path,
            token_stat_provider=provider,
            artifact_dir=args.artifact_dir,
            report_dir=report_dir,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Ablation setup error: {error}") from error
    except RuntimeError as error:
        raise SystemExit(
            f"Ablation runtime error: {error}. Check that the checkpoint is available and train.sh has already produced the processed dataset."
        ) from error

    print(f"model={config.model_source}")
    print(f"best_variant={summary['best_variant']}")
    print(f"json_report={summary['json_report_path']}")
    print(f"markdown_report={summary['markdown_report_path']}")


if __name__ == "__main__":
    main()
