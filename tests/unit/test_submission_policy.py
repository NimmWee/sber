from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from submission.frozen_best import score_private_frozen_submission, train_frozen_best_submission


class SubmissionPolicyTest(unittest.TestCase):
    def test_train_rejects_public_benchmark_as_training_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "knowledge_bench_public.csv"
            dataset_path.write_text("prompt,response,is_hallucination\nq,a,0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "public benchmark"):
                train_frozen_best_submission(
                    dataset_path=dataset_path,
                    token_stat_provider=object(),
                    artifact_dir=Path(temp_dir) / "artifacts",
                )

    def test_private_scoring_rejects_labeled_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "knowledge_bench_public.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["prompt", "response", "is_hallucination"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt": "Q",
                        "response": "A",
                        "is_hallucination": "1",
                    }
                )

            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                score_private_frozen_submission(
                    input_path=input_path,
                    output_path=Path(temp_dir) / "scores.csv",
                    token_stat_provider=object(),
                    bundle=object(),
                )


if __name__ == "__main__":
    unittest.main()
