from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.token_stats import TransformersTokenStatProvider
from submission.frozen_best import benchmark_frozen_submission_latency
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark latency of the frozen hallucination detector on prompt/response CSV rows."
    )
    parser.add_argument("--config", default=None, help="Path to provider config JSON.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="CSV with prompt and response columns.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "model" / "frozen_best"),
        help="Directory with frozen model artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory for latency reports.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Maximum number of rows to use for the latency benchmark.",
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
        summary = benchmark_frozen_submission_latency(
            dataset_path=args.dataset_path,
            token_stat_provider=provider,
            artifact_dir=args.artifact_dir,
            report_dir=report_dir,
            sample_size=args.max_samples,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"Latency benchmark setup error: {error}") from error
    except RuntimeError as error:
        raise SystemExit(
            f"Latency benchmark runtime error: {error}. Check that the checkpoint and dataset path are available in Kaggle."
        ) from error

    total_ms = summary.get("total_ms", {})
    avg_latency_ms = total_ms.get("avg", summary.get("avg_latency_ms"))
    p50_latency_ms = total_ms.get("p50")
    p95_latency_ms = total_ms.get("p95")
    p99_latency_ms = total_ms.get("p99")

    print(f"model={config.model_source}")
    if "sample_size" in summary:
        print(f"sample_size={summary['sample_size']}")
    if avg_latency_ms is not None:
        print(f"avg_latency_ms={avg_latency_ms:.3f}")
    if p50_latency_ms is not None:
        print(f"p50_latency_ms={p50_latency_ms:.3f}")
    if p95_latency_ms is not None:
        print(f"p95_latency_ms={p95_latency_ms:.3f}")
    if p99_latency_ms is not None:
        print(f"p99_latency_ms={p99_latency_ms:.3f}")
    print(f"json_report={summary['json_report_path']}")
    print(f"markdown_report={summary['markdown_report_path']}")


if __name__ == "__main__":
    main()
