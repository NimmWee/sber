from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.triviaqa_generation import (
    TriviaQAExample,
    build_generated_triviaqa_rows,
    load_triviaqa_examples,
    select_triviaqa_examples,
)


def test_load_triviaqa_examples_supports_official_json_shape(tmp_path) -> None:
    dataset_path = tmp_path / "triviaqa.json"
    dataset_path.write_text(
        json.dumps(
            {
                "Data": [
                    {
                        "Question": "Who wrote Hamlet?",
                        "Answer": {"Value": "William Shakespeare"},
                    },
                    {
                        "Question": "What is the capital of France?",
                        "Answer": {"Aliases": ["Paris", "City of Paris"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    examples = load_triviaqa_examples(dataset_path)

    assert [example.prompt for example in examples] == [
        "Who wrote Hamlet?",
        "What is the capital of France?",
    ]
    assert [example.reference_answer for example in examples] == [
        "William Shakespeare",
        "Paris",
    ]


def test_select_triviaqa_examples_is_deterministic(tmp_path) -> None:
    dataset_path = tmp_path / "triviaqa.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q3", "answer": "A3"}),
                json.dumps({"question": "Q1", "answer": "A1"}),
                json.dumps({"question": "Q2", "answer": "A2"}),
            ]
        ),
        encoding="utf-8",
    )
    examples = load_triviaqa_examples(dataset_path)

    selected_once = select_triviaqa_examples(examples, max_samples=2)
    selected_twice = select_triviaqa_examples(examples, max_samples=2)

    assert [example.prompt for example in selected_once] == ["Q3", "Q1"]
    assert [example.prompt for example in selected_twice] == ["Q3", "Q1"]


def test_build_generated_triviaqa_rows_preserves_output_schema() -> None:
    examples = [
        TriviaQAExample(
            prompt="Who wrote Hamlet?",
            reference_answer="William Shakespeare",
        )
    ]

    rows = build_generated_triviaqa_rows(
        examples=examples,
        generated_responses=["William Shakespeare wrote Hamlet."],
    )

    assert rows == [
        {
            "prompt": "Who wrote Hamlet?",
            "reference_answer": "William Shakespeare",
            "response": "William Shakespeare wrote Hamlet.",
            "source": "triviaqa",
        }
    ]


def test_load_triviaqa_examples_supports_pair_parquet_shape(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "train-00000-of-00001.parquet"
    dataset_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "data.triviaqa_generation._load_parquet_records",
        lambda _: [
            {
                "query": "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
                "answer": "Sinclair Lewis",
            }
        ],
    )

    examples = load_triviaqa_examples(dataset_path)

    assert examples == [
        TriviaQAExample(
            prompt="Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
            reference_answer="Sinclair Lewis",
        )
    ]
