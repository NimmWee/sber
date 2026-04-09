from pathlib import Path
import csv
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from features.extractor import InternalModelSignal, TokenUncertaintyStat
from submission.frozen_best import (
    FrozenSubmissionBundle,
    build_frozen_best_metadata,
    score_private_frozen_submission,
)


class _FakeHead:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features) -> float:
        return self.probability


def test_build_frozen_best_metadata_uses_historical_best_commit() -> None:
    metadata = build_frozen_best_metadata()

    assert metadata["historical_best_commit"] == "d3fa946"
    assert metadata["historical_best_variant"] == "baseline_plus_all_specialists"
    assert metadata["historical_best_pr_auc"] == 0.6881


def test_score_private_frozen_submission_writes_output_csv(tmp_path) -> None:
    input_path = tmp_path / "private.csv"
    output_path = tmp_path / "scores.csv"
    input_path.write_text(
        "prompt,response\nWho wrote Hamlet?,William Shakespeare wrote Hamlet.\n",
        encoding="utf-8",
    )

    class FakeProvider:
        def collect_signals(self, *, prompt: str, response: str):
            return type(
                "Collected",
                (),
                {
                    "token_stats": [
                        TokenUncertaintyStat(
                            token="Hamlet",
                            logprob=-0.2,
                            entropy=0.3,
                            top1_top2_margin=0.7,
                        )
                    ],
                    "internal_signal": InternalModelSignal(
                        last_layer_pooled_l2=1.0,
                        last_layer_pooled_mean_abs=0.5,
                        selected_layer_norm_variance=0.1,
                        layer_disagreement_mean=0.2,
                        selected_layer_disagreement_max=0.3,
                        early_late_layer_consistency=0.8,
                    ),
                },
            )()

    bundle = FrozenSubmissionBundle(
        baseline_head=_FakeHead(0.4),
        numeric_head=_FakeHead(0.6),
        entity_head=_FakeHead(0.2),
        long_head=_FakeHead(0.8),
        metadata=build_frozen_best_metadata(),
    )

    summary = score_private_frozen_submission(
        input_path=input_path,
        output_path=output_path,
        token_stat_provider=FakeProvider(),
        bundle=bundle,
    )

    assert summary["sample_size"] == 1
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["hallucination_probability"] != ""

