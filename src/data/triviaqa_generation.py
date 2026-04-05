from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TriviaQAExample:
    prompt: str
    reference_answer: str
    source: str = "triviaqa"


def load_triviaqa_examples(path: str | Path) -> list[TriviaQAExample]:
    dataset_path = Path(path)
    if dataset_path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        else:
            records = payload.get("Data", [])

    examples: list[TriviaQAExample] = []
    for record in records:
        prompt = _extract_question(record)
        reference_answer = _extract_answer(record)
        examples.append(
            TriviaQAExample(
                prompt=prompt.strip(),
                reference_answer=reference_answer.strip(),
            )
        )
    return examples


def select_triviaqa_examples(
    examples: list[TriviaQAExample],
    *,
    max_samples: int | None,
) -> list[TriviaQAExample]:
    if max_samples is None or max_samples >= len(examples):
        return list(examples)
    return list(examples[:max_samples])


def build_generated_triviaqa_rows(
    *,
    examples: list[TriviaQAExample],
    generated_responses: list[str],
) -> list[dict[str, str]]:
    if len(examples) != len(generated_responses):
        raise ValueError("examples and generated_responses must have the same length")

    return [
        {
            "prompt": example.prompt,
            "reference_answer": example.reference_answer,
            "response": response,
            "source": example.source,
        }
        for example, response in zip(examples, generated_responses)
    ]


def write_generated_triviaqa_rows(
    *,
    rows: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _extract_question(record: dict[str, Any]) -> str:
    question = record.get("question") or record.get("Question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("TriviaQA record must contain a non-empty question")
    return question


def _extract_answer(record: dict[str, Any]) -> str:
    answer = record.get("answer") or record.get("Answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    if isinstance(answer, dict):
        value = answer.get("Value")
        if isinstance(value, str) and value.strip():
            return value
        aliases = answer.get("Aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    return alias
    raise ValueError("TriviaQA record must contain a non-empty answer")
