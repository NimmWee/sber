from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.public_benchmark import load_public_benchmark_examples


def test_load_public_benchmark_examples_reads_prompt_response_and_labels(tmp_path) -> None:
    csv_path = tmp_path / "knowledge_bench_public.csv"
    csv_path.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\n"
        "Question A,Answer A,False,Reference A,Note A\n"
        "Question B,Answer B,True,Reference B,Note B\n",
        encoding="utf-8",
    )

    examples = load_public_benchmark_examples(csv_path)

    assert len(examples) == 2
    assert examples[0].prompt == "Question A"
    assert examples[0].response == "Answer A"
    assert examples[0].label == 0
    assert examples[1].label == 1


def test_load_public_benchmark_examples_fails_clearly_for_missing_columns(tmp_path) -> None:
    csv_path = tmp_path / "knowledge_bench_public.csv"
    csv_path.write_text(
        "prompt,response,label\nQuestion A,Answer A,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required columns"):
        load_public_benchmark_examples(csv_path)
