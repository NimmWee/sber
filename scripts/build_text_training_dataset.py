from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.textual_dataset import build_textual_training_dataset, export_textual_training_dataset
from eval.public_benchmark import load_public_benchmark_examples
from utils.script_helpers import resolve_public_benchmark_path, resolve_text_training_seed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=None)
    parser.add_argument("--public-benchmark-path", default=None)
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "processed" / "textual_training_dataset.jsonl"),
    )
    args = parser.parse_args()

    seed_path = resolve_text_training_seed_path(
        project_root=PROJECT_ROOT,
        explicit_seed_path=args.seed_path,
    )
    public_benchmark_path = resolve_public_benchmark_path(
        project_root=PROJECT_ROOT,
        explicit_dataset_path=args.public_benchmark_path,
    )
    dataset = build_textual_training_dataset(
        seed_path=seed_path,
        public_eval_examples=load_public_benchmark_examples(public_benchmark_path),
    )
    artifact_path = export_textual_training_dataset(dataset=dataset, output_path=args.output_path)

    print(f"seed_path={seed_path}")
    print(f"public_benchmark_path={public_benchmark_path}")
    print(f"sample_size={dataset.summary['sample_size']}")
    print(f"hallucination_count={dataset.summary['hallucination_count']}")
    print(f"non_hallucination_count={dataset.summary['non_hallucination_count']}")
    print(f"source_count={len(dataset.summary.get('source_name_distribution', {}))}")
    print(
        "hallucination_bucket_coverage="
        f"{dataset.summary.get('hallucination_bucket_coverage', {})}"
    )
    print(
        "difficulty_heuristics="
        f"{dataset.summary.get('difficulty_heuristics', {})}"
    )
    print(f"warnings={'; '.join(dataset.summary.get('warnings', []))}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
