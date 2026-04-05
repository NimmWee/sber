from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.triviaqa_labeling import label_triviaqa_generated_responses
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import resolve_transformers_provider_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--input-path",
        default=str(PROJECT_ROOT / "data" / "textual" / "triviaqa_generated_responses.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "textual" / "triviaqa_labeled.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = label_triviaqa_generated_responses(
        input_path=args.input_path,
        output_path=args.output_path,
        token_stat_provider=provider,
        batch_size=args.batch_size,
    )

    print(f"model={config.model_source}")
    print(f"total_labeled_samples={summary['total_labeled_samples']}")
    print(f"hallucination_count={summary['hallucination_count']}")
    print(f"non_hallucination_count={summary['non_hallucination_count']}")
    print(f"hallucination_ratio={summary['hallucination_ratio']:.4f}")
    print(f"output_path={summary['output_path']}")


if __name__ == "__main__":
    main()
