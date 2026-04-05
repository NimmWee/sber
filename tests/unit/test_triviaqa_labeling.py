from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.triviaqa_labeling import (
    build_labeled_triviaqa_rows,
    resolve_judge_labels,
    is_skippable_response,
    load_generated_triviaqa_rows,
    parse_judge_label,
    write_labeled_triviaqa_rows,
)


def test_parse_judge_label_accepts_only_zero_or_one() -> None:
    assert parse_judge_label("0") == 0
    assert parse_judge_label(" 1\n") == 1
    assert parse_judge_label("label: 1") is None
    assert parse_judge_label("2") is None


def test_is_skippable_response_flags_empty_and_idk_variants() -> None:
    assert is_skippable_response("")
    assert is_skippable_response(" I don't know. ")
    assert is_skippable_response("Sorry, I cannot answer that.")
    assert not is_skippable_response("Paris is the capital of France.")


def test_build_labeled_triviaqa_rows_preserves_schema() -> None:
    rows = build_labeled_triviaqa_rows(
        source_rows=[
            {
                "prompt": "Who wrote Hamlet?",
                "reference_answer": "William Shakespeare",
                "response": "William Shakespeare wrote Hamlet.",
                "source": "triviaqa",
            }
        ],
        labels=[0],
    )

    assert rows == [
        {
            "prompt": "Who wrote Hamlet?",
            "reference_answer": "William Shakespeare",
            "response": "William Shakespeare wrote Hamlet.",
            "label": 0,
            "source": "triviaqa",
        }
    ]


def test_write_labeled_triviaqa_rows_writes_jsonl_output(tmp_path) -> None:
    output_path = tmp_path / "triviaqa_labeled.jsonl"
    rows = [
        {
            "prompt": "Who wrote Hamlet?",
            "reference_answer": "William Shakespeare",
            "response": "William Shakespeare wrote Hamlet.",
            "label": 0,
            "source": "triviaqa",
        }
    ]

    artifact_path = write_labeled_triviaqa_rows(rows=rows, output_path=output_path)

    assert artifact_path == output_path
    assert output_path.read_text(encoding="utf-8").strip() == json.dumps(rows[0], ensure_ascii=False)


def test_load_generated_triviaqa_rows_reads_jsonl(tmp_path) -> None:
    input_path = tmp_path / "triviaqa_generated.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Who wrote Hamlet?",
                        "reference_answer": "William Shakespeare",
                        "response": "William Shakespeare wrote Hamlet.",
                        "source": "triviaqa",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_generated_triviaqa_rows(input_path)

    assert rows[0]["source"] == "triviaqa"


def test_resolve_judge_labels_retries_once_for_invalid_output() -> None:
    labels = resolve_judge_labels(
        first_pass_outputs=["maybe", "0"],
        retry_outputs=["1"],
    )

    assert labels == [1, 0]
