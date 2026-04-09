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

from submission.frozen_best import FrozenSubmissionBundle, score_private_frozen_submission


class _StubHead:
    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_proba(self, _features) -> float:
        return self._probability


class _StubProvider:
    def collect(self, prompt: str, response: str):
        return []


class BooleanScoringTest(unittest.TestCase):
    def test_boolean_mode_uses_threshold_0_3(self) -> None:
        bundle = FrozenSubmissionBundle(
            baseline_head=_StubHead(0.3),
            numeric_head=_StubHead(0.3),
            entity_head=_StubHead(0.3),
            long_head=_StubHead(0.3),
            metadata={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            output_path = Path(temp_dir) / "output.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
                writer.writeheader()
                writer.writerow({"prompt": "Q", "response": "A"})

            summary = score_private_frozen_submission(
                input_path=input_path,
                output_path=output_path,
                token_stat_provider=_StubProvider(),
                bundle=bundle,
                output_mode="boolean",
                label_threshold=0.3,
            )

            self.assertEqual(summary["sample_size"], 1)
            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader)
            self.assertEqual(row["hallucination"], "true")
            self.assertNotIn("hallucination_probability", row)


if __name__ == "__main__":
    unittest.main()
