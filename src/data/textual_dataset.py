from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Callable

from eval.runner import RawLabeledExample


NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
TITLECASE_PATTERN = re.compile(r"\b[A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*\b")
PLACE_PREPOSITION_PATTERN = re.compile(
    r"\b(?:in|at|from|to|near|around)\s+[A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*\b"
)

PERSON_SWAPS = {
    "Alexander Fleming": "Marie Curie",
    "Albert Einstein": "Isaac Newton",
    "Antonio Vivaldi": "Johann Sebastian Bach",
    "Nelson Mandela": "Desmond Tutu",
}
PLACE_SWAPS = {
    "Lima": "Cusco",
    "Peru": "Chile",
    "Berlin": "Munich",
    "Budapest": "Vienna",
    "Strasbourg": "Brussels",
    "Venice": "Genoa",
    "Machu Picchu": "Lake Titicaca",
}
ORGANIZATION_SWAPS = {
    "NASA": "NOAA",
    "Sony": "Nintendo",
    "The Nobel Foundation": "The United Nations",
    "Springer Nature": "Elsevier",
    "King's College, Cambridge": "Trinity College, Oxford",
    "European Space Agency": "European Southern Observatory",
}


JudgeDecision = tuple[bool, dict]
JudgeFilter = Callable[["TextualTrainingExample"], JudgeDecision]


@dataclass(frozen=True)
class PublicSeedRecord:
    prompt: str
    answer: str
    source_name: str
    provenance: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextualTrainingExample:
    prompt: str
    response: str
    label: int
    split: str
    source_type: str
    source_name: str
    provenance: str
    generation_method: str
    corruption_type: str | None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextualTrainingDataset:
    train_examples: list[TextualTrainingExample]
    dev_examples: list[TextualTrainingExample]
    summary: dict[str, object]


def load_public_seed_records(path: str | Path) -> list[PublicSeedRecord]:
    records: list[PublicSeedRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        _validate_seed_payload(payload)
        records.append(
            PublicSeedRecord(
                prompt=payload["prompt"],
                answer=payload["answer"],
                source_name=payload["source_name"],
                provenance=payload["provenance"],
                metadata=dict(payload.get("metadata", {})),
            )
        )
    if not records:
        raise ValueError("seed_path must contain at least one public seed record")
    return records


def build_textual_training_dataset(
    *,
    seed_path: str | Path,
    public_eval_examples: list[RawLabeledExample] | None = None,
    judge_filter: JudgeFilter | None = None,
) -> TextualTrainingDataset:
    seed_records = load_public_seed_records(seed_path)
    train_examples: list[TextualTrainingExample] = []
    dev_examples: list[TextualTrainingExample] = []
    all_examples: list[TextualTrainingExample] = []
    source_type_distribution: dict[str, int] = {}
    source_name_distribution: dict[str, int] = {}
    generation_distribution: dict[str, int] = {}
    corruption_taxonomy: dict[str, int] = {}
    bucket_label_counts = {
        "numbers": {"non_hallucination_count": 0, "hallucination_count": 0},
        "entity_like_tokens": {"non_hallucination_count": 0, "hallucination_count": 0},
        "places": {"non_hallucination_count": 0, "hallucination_count": 0},
        "long_responses": {"non_hallucination_count": 0, "hallucination_count": 0},
    }
    trivial_examples: list[dict[str, object]] = []

    example_index = 0
    for seed_index, seed in enumerate(seed_records):
        prompt_variants = _build_prompt_variants(seed.prompt)
        correct_variants = _build_correct_variants(seed)
        hallucinated_variants = _build_hallucinated_variants(seed)
        for prompt_variant in prompt_variants:
            for response_variant, generation_method, source_type in correct_variants:
                example = _build_textual_example(
                    prompt=prompt_variant,
                    response=response_variant,
                    label=0,
                    split="dev" if example_index % 5 == 0 else "train",
                    source_type=source_type,
                    source_name=seed.source_name,
                    provenance=seed.provenance,
                    generation_method=generation_method,
                    corruption_type=None,
                    seed_index=seed_index,
                    seed_metadata=seed.metadata,
                )
                if _apply_judge_filter(example, judge_filter):
                    _append_dataset_example(
                        example=example,
                        train_examples=train_examples,
                        dev_examples=dev_examples,
                        all_examples=all_examples,
                        source_type_distribution=source_type_distribution,
                        source_name_distribution=source_name_distribution,
                        generation_distribution=generation_distribution,
                        corruption_taxonomy=corruption_taxonomy,
                        bucket_label_counts=bucket_label_counts,
                    )
                    example_index += 1
            for response_variant, corruption_type, generation_method in hallucinated_variants:
                example = _build_textual_example(
                    prompt=prompt_variant,
                    response=response_variant,
                    label=1,
                    split="dev" if example_index % 5 == 0 else "train",
                    source_type="synthetic_corruption",
                    source_name=seed.source_name,
                    provenance=seed.provenance,
                    generation_method=generation_method,
                    corruption_type=corruption_type,
                    seed_index=seed_index,
                    seed_metadata=seed.metadata,
                )
                if _is_too_trivial(seed.answer, response_variant):
                    trivial_examples.append(
                        {
                            "prompt": prompt_variant,
                            "response": response_variant,
                            "corruption_type": corruption_type,
                        }
                    )
                if _apply_judge_filter(example, judge_filter):
                    _append_dataset_example(
                        example=example,
                        train_examples=train_examples,
                        dev_examples=dev_examples,
                        all_examples=all_examples,
                        source_type_distribution=source_type_distribution,
                        source_name_distribution=source_name_distribution,
                        generation_distribution=generation_distribution,
                        corruption_taxonomy=corruption_taxonomy,
                        bucket_label_counts=bucket_label_counts,
                    )
                    example_index += 1

    summary = _build_dataset_summary(
        examples=all_examples,
        source_type_distribution=source_type_distribution,
        source_name_distribution=source_name_distribution,
        generation_distribution=generation_distribution,
        corruption_taxonomy=corruption_taxonomy,
        bucket_label_counts=bucket_label_counts,
        trivial_examples=trivial_examples,
        public_eval_examples=public_eval_examples or [],
    )
    return TextualTrainingDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        summary=summary,
    )


def export_textual_training_dataset(
    *,
    dataset: TextualTrainingDataset,
    output_path: str | Path,
) -> Path:
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(example) for example in dataset.train_examples + dataset.dev_examples]
    artifact_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )
    return artifact_path


def load_textual_training_dataset(path: str | Path) -> list[TextualTrainingExample]:
    examples: list[TextualTrainingExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        examples.append(
            TextualTrainingExample(
                prompt=payload["prompt"],
                response=payload["response"],
                label=int(payload["label"]),
                split=payload["split"],
                source_type=payload["source_type"],
                source_name=payload["source_name"],
                provenance=payload["provenance"],
                generation_method=payload["generation_method"],
                corruption_type=payload.get("corruption_type"),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    if not examples:
        raise ValueError("textual training dataset must not be empty")
    return examples


def _build_textual_example(
    *,
    prompt: str,
    response: str,
    label: int,
    split: str,
    source_type: str,
    source_name: str,
    provenance: str,
    generation_method: str,
    corruption_type: str | None,
    seed_index: int,
    seed_metadata: dict[str, object],
) -> TextualTrainingExample:
    metadata = {
        "seed_index": seed_index,
        **dict(seed_metadata),
        "bucket_flags": _bucket_flags(response),
    }
    return TextualTrainingExample(
        prompt=prompt,
        response=response,
        label=label,
        split=split,
        source_type=source_type,
        source_name=source_name,
        provenance=provenance,
        generation_method=generation_method,
        corruption_type=corruption_type,
        metadata=metadata,
    )


def _append_dataset_example(
    *,
    example: TextualTrainingExample,
    train_examples: list[TextualTrainingExample],
    dev_examples: list[TextualTrainingExample],
    all_examples: list[TextualTrainingExample],
    source_type_distribution: dict[str, int],
    source_name_distribution: dict[str, int],
    generation_distribution: dict[str, int],
    corruption_taxonomy: dict[str, int],
    bucket_label_counts: dict[str, dict[str, int]],
) -> None:
    all_examples.append(example)
    if example.split == "dev":
        dev_examples.append(example)
    else:
        train_examples.append(example)
    source_type_distribution[example.source_type] = (
        source_type_distribution.get(example.source_type, 0) + 1
    )
    source_name_distribution[example.source_name] = (
        source_name_distribution.get(example.source_name, 0) + 1
    )
    generation_distribution[example.generation_method] = (
        generation_distribution.get(example.generation_method, 0) + 1
    )
    if example.corruption_type is not None:
        corruption_taxonomy[example.corruption_type] = (
            corruption_taxonomy.get(example.corruption_type, 0) + 1
        )
    for bucket_name, is_present in example.metadata["bucket_flags"].items():
        if bucket_name not in bucket_label_counts or not is_present:
            continue
        label_key = "hallucination_count" if example.label == 1 else "non_hallucination_count"
        bucket_label_counts[bucket_name][label_key] += 1


def _build_dataset_summary(
    *,
    examples: list[TextualTrainingExample],
    source_type_distribution: dict[str, int],
    source_name_distribution: dict[str, int],
    generation_distribution: dict[str, int],
    corruption_taxonomy: dict[str, int],
    bucket_label_counts: dict[str, dict[str, int]],
    trivial_examples: list[dict[str, object]],
    public_eval_examples: list[RawLabeledExample],
) -> dict[str, object]:
    hallucination_count = sum(example.label == 1 for example in examples)
    non_hallucination_count = sum(example.label == 0 for example in examples)
    duplicate_diagnostics = _duplicate_diagnostics(examples)
    leakage_checks = _leakage_checks(examples, public_eval_examples)
    bucket_coverage = {
        bucket_name: counts["hallucination_count"] + counts["non_hallucination_count"]
        for bucket_name, counts in bucket_label_counts.items()
    }
    bucket_label_ratios = {
        bucket_name: _bucket_label_ratio(counts)
        for bucket_name, counts in bucket_label_counts.items()
    }
    warnings = _dataset_warnings(
        bucket_label_counts=bucket_label_counts,
        duplicate_diagnostics=duplicate_diagnostics,
        trivial_examples=trivial_examples,
        leakage_checks=leakage_checks,
        source_name_distribution=source_name_distribution,
    )
    return {
        "sample_size": len(examples),
        "train_size": sum(example.split == "train" for example in examples),
        "dev_size": sum(example.split == "dev" for example in examples),
        "hallucination_count": hallucination_count,
        "non_hallucination_count": non_hallucination_count,
        "label_balance": {
            "hallucination_ratio": hallucination_count / len(examples) if examples else 0.0,
            "non_hallucination_ratio": non_hallucination_count / len(examples) if examples else 0.0,
        },
        "source_distribution": source_name_distribution,
        "source_name_distribution": source_name_distribution,
        "source_type_distribution": source_type_distribution,
        "generation_method_distribution": generation_distribution,
        "corruption_taxonomy": corruption_taxonomy,
        "bucket_coverage": bucket_coverage,
        "bucket_label_counts": bucket_label_counts,
        "bucket_label_ratios": bucket_label_ratios,
        "duplicate_count": duplicate_diagnostics["duplicate_count"],
        "near_duplicate_count": duplicate_diagnostics["near_duplicate_count"],
        "too_trivial_or_unrealistic_count": len(trivial_examples),
        "flagged_too_trivial_or_unrealistic_examples": trivial_examples,
        "leakage_checks": leakage_checks,
        "warnings": warnings,
    }


def _build_prompt_variants(prompt: str) -> tuple[str, ...]:
    stripped = prompt.rstrip(" ?.")
    return (
        prompt,
        f"Answer factually: {stripped}.",
    )


def _build_correct_variants(seed: PublicSeedRecord) -> list[tuple[str, str]]:
    variants: list[tuple[str, str, str]] = [
        (seed.answer, "source_seed_answer", "public_seed"),
        (f"The factual answer is: {seed.answer}", "concise_restatement", "correct_augmentation"),
    ]
    approximate_variant = _approximate_variant(seed.answer)
    if approximate_variant is not None:
        variants.append((approximate_variant, "approximate_correct_variant", "correct_augmentation"))
    long_variant = f"In a factual summary, {seed.answer}"
    if len(seed.answer.split()) < 18:
        long_variant = f"In a factual summary, {seed.answer} This answer stays grounded in the same public fact."
    variants.append((long_variant, "long_correct_variant", "correct_augmentation"))
    entity_dense_variant = _entity_dense_variant(seed.answer)
    if entity_dense_variant is not None:
        variants.append((entity_dense_variant, "entity_dense_correct_variant", "correct_augmentation"))
    return _dedupe_variant_rows(variants)


def _build_hallucinated_variants(
    seed: PublicSeedRecord,
) -> list[tuple[str, str, str]]:
    answer = seed.answer
    variants: list[tuple[str, str, str]] = []

    corrupted_year = _replace_year(answer)
    if corrupted_year is not None:
        variants.append((corrupted_year, "date_nearby", "nearby_year_corruption"))

    corrupted_number = _replace_number(answer)
    if corrupted_number is not None:
        variants.append((corrupted_number, "number_nearby", "nearby_number_corruption"))

    corrupted_person = _replace_from_lookup(answer, PERSON_SWAPS)
    if corrupted_person is not None:
        variants.append((corrupted_person, "entity_swap", "same_type_entity_swap"))

    corrupted_place = _replace_from_lookup(answer, PLACE_SWAPS)
    if corrupted_place is not None:
        variants.append((corrupted_place, "place_swap", "same_type_place_swap"))

    corrupted_org = _replace_from_lookup(answer, ORGANIZATION_SWAPS)
    if corrupted_org is not None:
        variants.append((corrupted_org, "organization_or_title_swap", "organization_title_swap"))

    if len(answer.split()) >= 16:
        local_corruption = _local_long_corruption(answer)
        if local_corruption is not None:
            variants.append((local_corruption, "local_fact_corruption", "long_local_corruption"))
        secondary_long_corruption = _secondary_long_corruption(answer)
        if secondary_long_corruption is not None:
            variants.append(
                (secondary_long_corruption, "local_fact_corruption", "long_secondary_local_corruption")
            )
    else:
        confident_short = _confident_short_corruption(answer)
        if confident_short is not None:
            variants.append((confident_short, "short_confident_wrong", "short_confident_corruption"))

    if not variants:
        fallback = f"{answer} This also claims an unsupported extra fact."
        variants.append((fallback, "unsupported_local_addition", "fallback_local_corruption"))
    return _dedupe_hallucinated_rows(variants)


def _apply_judge_filter(
    example: TextualTrainingExample,
    judge_filter: JudgeFilter | None,
) -> bool:
    if judge_filter is None:
        return True
    accepted, judge_metadata = judge_filter(example)
    example.metadata["judge"] = judge_metadata
    return accepted


def _approximate_variant(answer: str) -> str | None:
    match = NUMBER_PATTERN.search(answer)
    if match is None:
        return None
    number_text = match.group(0)
    replacement = f"about {number_text}" if "." not in number_text else f"roughly {number_text}"
    return answer[: match.start()] + replacement + answer[match.end() :]


def _entity_dense_variant(answer: str) -> str | None:
    if len(TITLECASE_PATTERN.findall(answer)) < 1:
        return None
    return f"Historically and factually, {answer}"


def _replace_year(answer: str) -> str | None:
    match = YEAR_PATTERN.search(answer)
    if match is None:
        return None
    year = int(match.group(0))
    replacement = str(year + 2 if year < 2024 else year - 2)
    return answer[: match.start()] + replacement + answer[match.end() :]


def _replace_number(answer: str) -> str | None:
    match = NUMBER_PATTERN.search(answer)
    if match is None:
        return None
    value = match.group(0)
    if YEAR_PATTERN.fullmatch(value):
        return None
    number = float(value)
    replacement = str(int(number + 2)) if number.is_integer() else f"{number + 0.5:.1f}"
    return answer[: match.start()] + replacement + answer[match.end() :]


def _replace_from_lookup(answer: str, lookup: dict[str, str]) -> str | None:
    for source_value, replacement in lookup.items():
        if source_value in answer:
            return answer.replace(source_value, replacement, 1)
    return None


def _local_long_corruption(answer: str) -> str | None:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
    if not sentences:
        return None
    first_sentence = sentences[0]
    corrupted = _replace_year(first_sentence) or _replace_number(first_sentence) or _replace_from_lookup(first_sentence, PERSON_SWAPS) or _replace_from_lookup(first_sentence, PLACE_SWAPS) or _replace_from_lookup(first_sentence, ORGANIZATION_SWAPS)
    if corrupted is None:
        corrupted = f"{first_sentence} It also incorrectly attributes one key fact."
    sentences[0] = corrupted
    return " ".join(sentences)


def _confident_short_corruption(answer: str) -> str | None:
    corrupted = _replace_year(answer) or _replace_number(answer) or _replace_from_lookup(answer, PERSON_SWAPS) or _replace_from_lookup(answer, PLACE_SWAPS) or _replace_from_lookup(answer, ORGANIZATION_SWAPS)
    if corrupted is None:
        return None
    return f"The correct answer is definitely: {corrupted}"


def _secondary_long_corruption(answer: str) -> str | None:
    corrupted = _replace_from_lookup(answer, PERSON_SWAPS) or _replace_from_lookup(answer, PLACE_SWAPS) or _replace_from_lookup(answer, ORGANIZATION_SWAPS) or _replace_number(answer) or _replace_year(answer)
    if corrupted is None or corrupted == answer:
        return None
    return corrupted


def _dedupe_variant_rows(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    deduped: list[tuple[str, str, str]] = []
    for response, generation_method, source_type in rows:
        normalized = " ".join(response.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((normalized, generation_method, source_type))
    return deduped


def _dedupe_hallucinated_rows(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    deduped: list[tuple[str, str, str]] = []
    for response, corruption_type, generation_method in rows:
        normalized = " ".join(response.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((normalized, corruption_type, generation_method))
    return deduped


def _is_too_trivial(reference: str, candidate: str) -> bool:
    return SequenceMatcher(a=reference, b=candidate).ratio() > 0.992


def _duplicate_diagnostics(examples: list[TextualTrainingExample]) -> dict[str, int]:
    exact = set()
    duplicate_count = 0
    near_duplicate_count = 0
    for index, left in enumerate(examples):
        fingerprint = (left.prompt, left.response, left.label)
        if fingerprint in exact:
            duplicate_count += 1
        exact.add(fingerprint)
        for right in examples[index + 1 :]:
            if left.label != right.label:
                continue
            if left.prompt != right.prompt:
                continue
            if SequenceMatcher(a=left.response, b=right.response).ratio() > 0.985:
                near_duplicate_count += 1
    return {
        "duplicate_count": duplicate_count,
        "near_duplicate_count": near_duplicate_count,
    }


def _leakage_checks(
    examples: list[TextualTrainingExample],
    public_eval_examples: list[RawLabeledExample],
) -> dict[str, int]:
    dataset_pairs = {(example.prompt, example.response) for example in examples}
    public_pairs = {(example.prompt, example.response) for example in public_eval_examples}
    dataset_prompts = {example.prompt for example in examples}
    public_prompts = {example.prompt for example in public_eval_examples}
    dataset_responses = {example.response for example in examples}
    public_responses = {example.response for example in public_eval_examples}
    return {
        "public_exact_example_overlap_count": len(dataset_pairs & public_pairs),
        "public_prompt_overlap_count": len(dataset_prompts & public_prompts),
        "public_response_overlap_count": len(dataset_responses & public_responses),
    }


def _bucket_flags(text: str) -> dict[str, bool]:
    token_count = len(re.findall(r"\w+", text))
    return {
        "numbers": bool(NUMBER_PATTERN.search(text) or YEAR_PATTERN.search(text)),
        "entity_like_tokens": len(TITLECASE_PATTERN.findall(text)) >= 2,
        "places": PLACE_PREPOSITION_PATTERN.search(text) is not None,
        "long_responses": token_count > 24,
    }


def _bucket_label_ratio(counts: dict[str, int]) -> dict[str, float]:
    total = counts["hallucination_count"] + counts["non_hallucination_count"]
    if total == 0:
        return {"hallucination_ratio": 0.0, "non_hallucination_ratio": 0.0}
    return {
        "hallucination_ratio": counts["hallucination_count"] / total,
        "non_hallucination_ratio": counts["non_hallucination_count"] / total,
    }


def _dataset_warnings(
    *,
    bucket_label_counts: dict[str, dict[str, int]],
    duplicate_diagnostics: dict[str, int],
    trivial_examples: list[dict[str, object]],
    leakage_checks: dict[str, int],
    source_name_distribution: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    for bucket_name, counts in bucket_label_counts.items():
        total = counts["hallucination_count"] + counts["non_hallucination_count"]
        if total == 0:
            warnings.append(f"{bucket_name} bucket has no coverage")
            continue
        hallucination_ratio = counts["hallucination_count"] / total
        if hallucination_ratio < 0.2:
            warnings.append(f"{bucket_name} bucket is underrepresented on hallucination side")
        if hallucination_ratio > 0.8:
            warnings.append(f"{bucket_name} bucket is overly dominated by hallucinations")
        if bucket_name == "long_responses" and total < 12:
            warnings.append("long-response coverage is too low")
        if bucket_name == "numbers" and counts["hallucination_count"] < 6:
            warnings.append("numeric hallucination coverage is too low")
        if bucket_name == "entity_like_tokens" and counts["hallucination_count"] < 6:
            warnings.append("entity hallucination coverage is too low")
    if duplicate_diagnostics["near_duplicate_count"] > 8:
        warnings.append("near-duplicate count is high")
    if trivial_examples:
        warnings.append("some synthetic examples are too trivial or unrealistic")
    if leakage_checks["public_exact_example_overlap_count"] > 0:
        warnings.append("public benchmark exact overlap detected")
    total_examples = sum(source_name_distribution.values())
    if total_examples:
        dominant_source_share = max(source_name_distribution.values()) / total_examples
        if dominant_source_share > 0.7:
            warnings.append("one source dominates too much")
    return warnings


def _validate_seed_payload(payload: dict[str, object]) -> None:
    required_fields = {"prompt", "answer", "source_name", "provenance"}
    missing_fields = sorted(required_fields - set(payload.keys()))
    if missing_fields:
        raise ValueError(f"seed record is missing required fields: {', '.join(missing_fields)}")
    metadata = payload.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("seed record metadata must be a mapping if provided")
