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
    elif dataset_path.suffix == ".parquet":
        records = _load_parquet_records(dataset_path)
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
    validate: bool = False,
) -> Path:
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if validate:
        diagnostics = validate_jsonl_rows(artifact_path)
        if diagnostics["invalid_row_count"] > 0:
            raise ValueError("generated TriviaQA JSONL contains invalid rows")
    return artifact_path


def validate_jsonl_rows(path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid_row_count += 1
            continue
        rows.append(row)
    return {
        "valid_row_count": len(rows),
        "invalid_row_count": invalid_row_count,
        "rows": rows,
    }


def clean_invalid_jsonl_rows(
    *,
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    diagnostics = validate_jsonl_rows(input_path)
    write_generated_triviaqa_rows(
        rows=diagnostics["rows"],
        output_path=output_path,
        validate=True,
    )
    return {
        "kept_row_count": diagnostics["valid_row_count"],
        "dropped_row_count": diagnostics["invalid_row_count"],
        "output_path": str(output_path),
    }


def _extract_question(record: dict[str, Any]) -> str:
    question = record.get("question") or record.get("Question") or record.get("query")
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


def _load_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path).to_dict(orient="records")
    except ModuleNotFoundError:
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq.read_table(path).to_pylist()
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "parquet support requires pandas or pyarrow to load TriviaQA locally"
        ) from error
