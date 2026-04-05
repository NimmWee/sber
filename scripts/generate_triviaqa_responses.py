from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.triviaqa_generation import (
    build_generated_triviaqa_rows,
    load_triviaqa_examples,
    select_triviaqa_examples,
    write_generated_triviaqa_rows,
)
from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import resolve_transformers_provider_config, resolve_triviaqa_path


def generate_triviaqa_responses(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    token_stat_provider: TransformersTokenStatProvider,
    max_samples: int | None,
    batch_size: int,
) -> dict[str, Any]:
    examples = select_triviaqa_examples(
        load_triviaqa_examples(dataset_path),
        max_samples=max_samples,
    )
    responses = _generate_responses_for_examples(
        examples=examples,
        token_stat_provider=token_stat_provider,
        batch_size=batch_size,
    )
    rows = build_generated_triviaqa_rows(
        examples=examples,
        generated_responses=responses,
    )
    artifact_path = write_generated_triviaqa_rows(rows=rows, output_path=output_path)
    average_response_length = (
        sum(len(row["response"]) for row in rows) / len(rows) if rows else 0.0
    )
    return {
        "processed_samples": len(rows),
        "output_path": str(artifact_path),
        "average_response_length": average_response_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "textual" / "triviaqa_generated_responses.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    dataset_path = resolve_triviaqa_path(
        project_root=PROJECT_ROOT,
        explicit_dataset_path=args.dataset_path,
    )
    provider = TransformersTokenStatProvider(config=config)
    summary = generate_triviaqa_responses(
        dataset_path=dataset_path,
        output_path=args.output_path,
        token_stat_provider=provider,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    print(f"model={config.model_source}")
    print(f"processed_samples={summary['processed_samples']}")
    print(f"output_path={summary['output_path']}")
    print(f"average_response_length={summary['average_response_length']:.2f}")


def _generate_responses_for_examples(
    *,
    examples: list[Any],
    token_stat_provider: TransformersTokenStatProvider,
    batch_size: int,
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tokenizer = token_stat_provider._get_tokenizer()
    model = token_stat_provider._get_model()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(tokenizer, "pad_token", None) is None and eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    responses: list[str] = []
    for batch_start in range(0, len(examples), batch_size):
        batch_examples = examples[batch_start : batch_start + batch_size]
        generation_prompts = [
            f"{example.prompt}{token_stat_provider.config.response_delimiter}"
            for example in batch_examples
        ]
        encoded = tokenizer(
            generation_prompts,
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
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            generated_tokens = generated[row_index, int(prompt_length) :]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(decoded)

    return responses


if __name__ == "__main__":
    main()
