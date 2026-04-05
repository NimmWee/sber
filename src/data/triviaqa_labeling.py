from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch

from inference.token_stats import TransformersTokenStatProvider


IDK_PATTERNS = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not sure",
    "unknown",
)


def load_generated_triviaqa_rows(path: str | Path) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_judge_label(output: str) -> int | None:
    match = re.fullmatch(r"\s*([01])\s*", output)
    if match is None:
        return None
    return int(match.group(1))


def is_skippable_response(response: str) -> bool:
    normalized = response.strip().lower()
    if normalized == "":
        return True
    return any(pattern in normalized for pattern in IDK_PATTERNS)


def resolve_judge_labels(
    *,
    first_pass_outputs: list[str],
    retry_outputs: list[str],
) -> list[int]:
    labels: list[int] = []
    retry_index = 0
    for output in first_pass_outputs:
        parsed = parse_judge_label(output)
        if parsed is not None:
            labels.append(parsed)
            continue
        if retry_index >= len(retry_outputs):
            raise ValueError("judge output was invalid after retry")
        retried = parse_judge_label(retry_outputs[retry_index])
        retry_index += 1
        if retried is None:
            raise ValueError("judge output was invalid after retry")
        labels.append(retried)
    return labels


def build_labeled_triviaqa_rows(
    *,
    source_rows: list[dict[str, str]],
    labels: list[int],
) -> list[dict[str, Any]]:
    if len(source_rows) != len(labels):
        raise ValueError("source_rows and labels must have the same length")

    return [
        {
            "prompt": row["prompt"],
            "reference_answer": row["reference_answer"],
            "response": row["response"],
            "label": label,
            "source": row["source"],
        }
        for row, label in zip(source_rows, labels)
    ]


def write_labeled_triviaqa_rows(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def label_triviaqa_generated_responses(
    *,
    input_path: str | Path,
    output_path: str | Path,
    token_stat_provider: TransformersTokenStatProvider,
    batch_size: int,
) -> dict[str, Any]:
    rows = load_generated_triviaqa_rows(input_path)
    candidate_rows = [row for row in rows if not is_skippable_response(row["response"])]
    judge_prompts = [_build_judge_prompt(row) for row in candidate_rows]
    first_pass_outputs = _generate_judge_outputs(
        prompts=judge_prompts,
        token_stat_provider=token_stat_provider,
        batch_size=batch_size,
        max_new_tokens=2,
    )

    retry_rows = [
        row
        for row, output in zip(candidate_rows, first_pass_outputs)
        if parse_judge_label(output) is None
    ]
    retry_prompts = [_build_retry_judge_prompt(row) for row in retry_rows]
    retry_outputs = (
        _generate_judge_outputs(
            prompts=retry_prompts,
            token_stat_provider=token_stat_provider,
            batch_size=1,
            max_new_tokens=2,
        )
        if retry_prompts
        else []
    )
    labels = resolve_judge_labels(
        first_pass_outputs=first_pass_outputs,
        retry_outputs=retry_outputs,
    )
    labeled_rows = build_labeled_triviaqa_rows(
        source_rows=candidate_rows,
        labels=labels,
    )
    artifact_path = write_labeled_triviaqa_rows(rows=labeled_rows, output_path=output_path)

    hallucination_count = sum(row["label"] for row in labeled_rows)
    non_hallucination_count = len(labeled_rows) - hallucination_count
    hallucination_ratio = (
        hallucination_count / len(labeled_rows) if labeled_rows else 0.0
    )
    return {
        "total_labeled_samples": len(labeled_rows),
        "hallucination_count": hallucination_count,
        "non_hallucination_count": non_hallucination_count,
        "hallucination_ratio": hallucination_ratio,
        "output_path": str(artifact_path),
    }


def _build_judge_prompt(row: dict[str, str]) -> str:
    return (
        "You are a strict factual judge.\n"
        "Compare the candidate response to the question and the reference answer.\n"
        "Output 1 if the candidate contains any factual error, contradiction, or unsupported factual claim.\n"
        "Output 0 only if the candidate is factually correct and consistent.\n"
        "Return exactly one character: 0 or 1.\n\n"
        f"Question: {row['prompt']}\n"
        f"Reference answer: {row['reference_answer']}\n"
        f"Candidate response: {row['response']}\n\n"
        "Judgment:"
    )


def _build_retry_judge_prompt(row: dict[str, str]) -> str:
    return (
        "Return exactly one character: 0 or 1.\n"
        "1 means the candidate has any factual error.\n"
        "0 means the candidate is factually correct.\n\n"
        f"Question: {row['prompt']}\n"
        f"Reference answer: {row['reference_answer']}\n"
        f"Candidate response: {row['response']}\n\n"
        "Answer with only 0 or 1:"
    )


def _generate_judge_outputs(
    *,
    prompts: list[str],
    token_stat_provider: TransformersTokenStatProvider,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts:
        return []

    tokenizer = token_stat_provider._get_tokenizer()
    model = token_stat_provider._get_model()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(tokenizer, "pad_token", None) is None and eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    outputs: list[str] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_device = token_stat_provider._input_device_for_model(model)
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded["attention_mask"].to(input_device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            generated_tokens = generated[row_index, int(prompt_length) :]
            outputs.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))
    return outputs
