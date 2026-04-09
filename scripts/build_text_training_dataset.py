from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.textual_dataset import build_textual_training_dataset, export_textual_training_dataset
from eval.runner import RawLabeledExample
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
        public_eval_examples=_load_public_benchmark_examples(public_benchmark_path),
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


def _load_public_benchmark_examples(dataset_path: str | Path) -> list[RawLabeledExample]:
    dataset_root = Path(dataset_path)
    with dataset_root.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    examples: list[RawLabeledExample] = []
    for row in rows:
        label_value = row.get("label")
        if label_value is None:
            label_value = row.get("is_hallucination")
        if label_value is None:
            raise ValueError("knowledge_bench_public.csv must contain label or is_hallucination")

        response_value = row.get("response") or row.get("model_answer")
        if response_value is None:
            raise ValueError("knowledge_bench_public.csv must contain response or model_answer")

        examples.append(
            RawLabeledExample(
                prompt=row["prompt"],
                response=response_value,
                label=_parse_public_benchmark_label(label_value),
            )
        )
    return examples


def _parse_public_benchmark_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"unsupported public benchmark label value: {value}")


if __name__ == "__main__":
    main()
