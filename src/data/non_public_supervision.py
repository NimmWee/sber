from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from eval.runner import RawLabeledExample


@dataclass(frozen=True)
class NonPublicSupervisionDataset:
    train_examples: list[RawLabeledExample]
    dev_examples: list[RawLabeledExample]
    summary: dict


@dataclass(frozen=True)
class SeedFact:
    prompt: str
    positive_response: str
    negative_response: str
    corruption_type: str
    is_long_response: bool = False


SEED_FACTS: tuple[SeedFact, ...] = (
    SeedFact(
        prompt="How many players can each soccer team have on the field during normal play?",
        positive_response="A standard soccer team has 11 players on the field during normal play.",
        negative_response="A standard soccer team has 10 players on the field during normal play.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="How many bones are in the typical adult human body?",
        positive_response="The typical adult human body has 206 bones.",
        negative_response="The typical adult human body has 208 bones.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="How many stripes are on the flag of the United States?",
        positive_response="The flag of the United States has 13 stripes.",
        negative_response="The flag of the United States has 12 stripes.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="At what temperature does pure water freeze in degrees Celsius?",
        positive_response="Pure water freezes at 0 degrees Celsius.",
        negative_response="Pure water freezes at 4 degrees Celsius.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="How many sides does a regular hexagon have?",
        positive_response="A regular hexagon has 6 sides.",
        negative_response="A regular hexagon has 8 sides.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="How many justices serve on the Supreme Court of the United States?",
        positive_response="The Supreme Court of the United States has 9 justices.",
        negative_response="The Supreme Court of the United States has 8 justices.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="In what year was the first iPhone released?",
        positive_response="The first iPhone was released in 2007.",
        negative_response="The first iPhone was released in 2008.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="In what year did the Titanic sink?",
        positive_response="The Titanic sank in 1912.",
        negative_response="The Titanic sank in 1914.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="In what year did the French Revolution begin?",
        positive_response="The French Revolution began in 1789.",
        negative_response="The French Revolution began in 1791.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="In what year did the World Wide Web become publicly available?",
        positive_response="The World Wide Web became publicly available in 1991.",
        negative_response="The World Wide Web became publicly available in 1993.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="In what year was the United States Declaration of Independence signed?",
        positive_response="The United States Declaration of Independence was signed in 1776.",
        negative_response="The United States Declaration of Independence was signed in 1778.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="In what year did Nelson Mandela become president of South Africa?",
        positive_response="Nelson Mandela became president of South Africa in 1994.",
        negative_response="Nelson Mandela became president of South Africa in 1996.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="Who developed the theory of relativity?",
        positive_response="Albert Einstein developed the theory of relativity.",
        negative_response="Isaac Newton developed the theory of relativity.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Who composed The Four Seasons?",
        positive_response="Antonio Vivaldi composed The Four Seasons.",
        negative_response="Johann Sebastian Bach composed The Four Seasons.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Who was the first woman to fly solo across the Atlantic Ocean?",
        positive_response="Amelia Earhart was the first woman to fly solo across the Atlantic Ocean.",
        negative_response="Bessie Coleman was the first woman to fly solo across the Atlantic Ocean.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Who wrote One Hundred Years of Solitude?",
        positive_response="Gabriel Garcia Marquez wrote One Hundred Years of Solitude.",
        negative_response="Jorge Luis Borges wrote One Hundred Years of Solitude.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Who formulated the laws of planetary motion?",
        positive_response="Johannes Kepler formulated the laws of planetary motion.",
        negative_response="Galileo Galilei formulated the laws of planetary motion.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Who discovered radium together with Pierre Curie?",
        positive_response="Marie Curie discovered radium together with Pierre Curie.",
        negative_response="Irene Joliot-Curie discovered radium together with Pierre Curie.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="Which river flows through Budapest?",
        positive_response="The Danube flows through Budapest.",
        negative_response="The Rhine flows through Budapest.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which country is home to Machu Picchu?",
        positive_response="Machu Picchu is in Peru.",
        negative_response="Machu Picchu is in Bolivia.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which Italian city is known as the City of Canals?",
        positive_response="Venice is known as the City of Canals in Italy.",
        negative_response="Genoa is known as the City of Canals in Italy.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which desert covers much of northern Africa?",
        positive_response="The Sahara covers much of northern Africa.",
        negative_response="The Kalahari covers much of northern Africa.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which sea lies between northeastern Africa and the Arabian Peninsula?",
        positive_response="The Red Sea lies between northeastern Africa and the Arabian Peninsula.",
        negative_response="The Arabian Sea lies between northeastern Africa and the Arabian Peninsula.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which city is home to the Brandenburg Gate?",
        positive_response="The Brandenburg Gate is in Berlin.",
        negative_response="The Brandenburg Gate is in Munich.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Which organization leads the James Webb Space Telescope mission?",
        positive_response="NASA leads the James Webb Space Telescope mission with international partners.",
        negative_response="NOAA leads the James Webb Space Telescope mission with international partners.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Which company created the PlayStation game console brand?",
        positive_response="Sony created the PlayStation game console brand.",
        negative_response="Nintendo created the PlayStation game console brand.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Which organization awards the Nobel Prizes?",
        positive_response="The Nobel Foundation awards the Nobel Prizes.",
        negative_response="The United Nations awards the Nobel Prizes.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Which title did Catherine the Great hold in Russia?",
        positive_response="Catherine the Great held the title Empress of Russia.",
        negative_response="Catherine the Great held the title Queen of Russia.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Which company publishes the journal Nature?",
        positive_response="Springer Nature publishes the journal Nature.",
        negative_response="Elsevier publishes the journal Nature.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Which university did Alan Turing attend for mathematics?",
        positive_response="Alan Turing studied mathematics at King's College, Cambridge.",
        negative_response="Alan Turing studied mathematics at Trinity College, Oxford.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Summarize the Voyager 1 mission milestone.",
        positive_response=(
            "Voyager 1 was launched in 1977 by NASA, completed major observations of Jupiter and Saturn, "
            "and later became the first spacecraft confirmed to enter interstellar space."
        ),
        negative_response=(
            "Voyager 1 was launched in 1981 by NASA, completed major observations of Jupiter and Saturn, "
            "and later became the first spacecraft confirmed to enter interstellar space."
        ),
        corruption_type="date_nearby",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Explain Alexander Fleming's role in the history of medicine.",
        positive_response=(
            "Alexander Fleming discovered penicillin in 1928, and that discovery later transformed medicine "
            "by enabling widely effective antibiotic treatment."
        ),
        negative_response=(
            "Alexander Fleming discovered penicillin in 1938, and that discovery later transformed medicine "
            "by enabling widely effective antibiotic treatment."
        ),
        corruption_type="date_nearby",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Explain who completed the first successful ascent of Mount Everest.",
        positive_response=(
            "Edmund Hillary and Tenzing Norgay completed the first confirmed successful ascent of Mount Everest "
            "in 1953 during a British expedition."
        ),
        negative_response=(
            "George Mallory and Tenzing Norgay completed the first confirmed successful ascent of Mount Everest "
            "in 1953 during a British expedition."
        ),
        corruption_type="entity_swap",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Describe the official seat of the European Parliament.",
        positive_response=(
            "The European Parliament has its official seat in Strasbourg, although additional parliamentary work "
            "also takes place in Brussels and Luxembourg."
        ),
        negative_response=(
            "The European Parliament has its official seat in Brussels, although additional parliamentary work "
            "also takes place in Brussels and Luxembourg."
        ),
        corruption_type="place_swap",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Summarize the Human Genome Project outcome.",
        positive_response=(
            "The Human Genome Project produced a reference sequence covering roughly 3 billion base pairs, "
            "providing a foundational map for later genomic research."
        ),
        negative_response=(
            "The Human Genome Project produced a reference sequence covering roughly 2 billion base pairs, "
            "providing a foundational map for later genomic research."
        ),
        corruption_type="number_nearby",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Summarize which agencies operate the Hubble Space Telescope.",
        positive_response=(
            "The Hubble Space Telescope is operated by NASA with major partnership support from the European Space Agency, "
            "and its science operations have produced landmark astronomical observations."
        ),
        negative_response=(
            "The Hubble Space Telescope is operated by the European Southern Observatory with major partnership support from the European Space Agency, "
            "and its science operations have produced landmark astronomical observations."
        ),
        corruption_type="organization_or_title_swap",
        is_long_response=True,
    ),
)


def build_non_public_supervision_dataset(
    *,
    public_eval_examples: list[RawLabeledExample] | None = None,
) -> NonPublicSupervisionDataset:
    examples: list[tuple[RawLabeledExample, str, bool]] = []
    flagged_examples: list[dict[str, str]] = []

    for seed in SEED_FACTS:
        prompt_variants = _build_prompt_variants(seed.prompt)
        correct_response_variants = _build_correct_response_variants(seed)
        hallucinated_response_variants = _build_hallucinated_response_variants(seed)
        if _is_too_trivial_or_unrealistic(seed):
            flagged_examples.append(
                {
                    "prompt": seed.prompt,
                    "negative_response": seed.negative_response,
                    "corruption_type": seed.corruption_type,
                }
            )
        for prompt_variant in prompt_variants:
            for response_variant in correct_response_variants:
                examples.append(
                    (
                        RawLabeledExample(
                            prompt=prompt_variant,
                            response=response_variant,
                            label=0,
                        ),
                        "clean_positive",
                        seed.is_long_response,
                    )
                )
            for response_variant in hallucinated_response_variants:
                examples.append(
                    (
                        RawLabeledExample(
                            prompt=prompt_variant,
                            response=response_variant,
                            label=1,
                        ),
                        seed.corruption_type,
                        seed.is_long_response,
                    )
                )

    train_examples: list[RawLabeledExample] = []
    dev_examples: list[RawLabeledExample] = []
    corruption_taxonomy: dict[str, int] = {
        "number_nearby": 0,
        "date_nearby": 0,
        "entity_swap": 0,
        "place_swap": 0,
        "organization_or_title_swap": 0,
    }
    long_response_count = 0
    positive_count = 0
    negative_count = 0
    number_heavy_count = 0
    entity_heavy_count = 0
    place_rich_count = 0
    approximate_or_range_style_count = 0
    risky_bucket_negative_coverage = {
        "numbers": 0,
        "entity_like_tokens": 0,
        "places": 0,
        "long_responses": 0,
    }
    risky_bucket_positive_coverage = {
        "numbers": 0,
        "entity_like_tokens": 0,
        "places": 0,
        "long_responses": 0,
    }

    for index, (example, corruption_type, is_long_response) in enumerate(examples):
        if example.label == 0:
            positive_count += 1
        else:
            negative_count += 1
            corruption_taxonomy[corruption_type] += 1
        if is_long_response:
            long_response_count += 1
        bucket_flags = _bucket_flags(example.response)
        if bucket_flags["numbers"]:
            number_heavy_count += 1
        if bucket_flags["entity_like_tokens"]:
            entity_heavy_count += 1
        if bucket_flags["places"]:
            place_rich_count += 1
        if bucket_flags["approximate_or_range_style"]:
            approximate_or_range_style_count += 1
        coverage_target = (
            risky_bucket_negative_coverage if example.label == 0 else risky_bucket_positive_coverage
        )
        for bucket_name in ("numbers", "entity_like_tokens", "places", "long_responses"):
            if bucket_flags[bucket_name]:
                coverage_target[bucket_name] += 1

        if index % 5 == 0:
            dev_examples.append(example)
        else:
            train_examples.append(example)

    duplicate_and_near_duplicate = _duplicate_diagnostics(train_examples + dev_examples)
    warnings = _dataset_warnings(
        non_hallucination_count=positive_count,
        hallucination_count=negative_count,
        risky_bucket_negative_coverage=risky_bucket_negative_coverage,
        risky_bucket_positive_coverage=risky_bucket_positive_coverage,
        too_trivial_or_unrealistic_count=len(flagged_examples),
        sample_size=len(train_examples) + len(dev_examples),
    )
    leakage_checks = _compute_leakage_checks(
        supervision_examples=train_examples + dev_examples,
        public_eval_examples=public_eval_examples or [],
    )
    summary = {
        "sample_size": len(train_examples) + len(dev_examples),
        "train_size": len(train_examples),
        "dev_size": len(dev_examples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "non_hallucination_count": positive_count,
        "hallucination_count": negative_count,
        "label_balance": {
            "positive_ratio": positive_count / (positive_count + negative_count),
            "negative_ratio": negative_count / (positive_count + negative_count),
            "non_hallucination_ratio": positive_count / (positive_count + negative_count),
            "hallucination_ratio": negative_count / (positive_count + negative_count),
        },
        "long_response_count": long_response_count,
        "number_heavy_count": number_heavy_count,
        "entity_heavy_count": entity_heavy_count,
        "place_rich_count": place_rich_count,
        "approximate_or_range_style_count": approximate_or_range_style_count,
        "risky_bucket_negative_coverage": risky_bucket_negative_coverage,
        "risky_bucket_positive_coverage": risky_bucket_positive_coverage,
        "corruption_taxonomy": corruption_taxonomy,
        "too_trivial_or_unrealistic_count": len(flagged_examples),
        "flagged_too_trivial_or_unrealistic_examples": flagged_examples,
        "duplicate_count": duplicate_and_near_duplicate["duplicate_count"],
        "near_duplicate_count": duplicate_and_near_duplicate["near_duplicate_count"],
        "warnings": warnings,
        "leakage_checks": leakage_checks,
    }
    return NonPublicSupervisionDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        summary=summary,
    )


def _build_prompt_variants(base_prompt: str) -> tuple[str, ...]:
    stripped = base_prompt.rstrip(" ?.")
    return (
        base_prompt,
        f"Give a factual answer only: {stripped}.",
    )


def _build_correct_response_variants(seed: SeedFact) -> tuple[str, ...]:
    variants = [
        seed.positive_response,
        f"The correct factual answer is: {seed.positive_response}",
    ]
    if seed.is_long_response:
        variants.append(f"In a concise factual summary, {seed.positive_response}")
    else:
        variants.append(f"For a precise answer, {seed.positive_response}")

    approximate_variant = _make_approximate_or_range_variant(seed.positive_response)
    if approximate_variant is not None:
        variants.append(approximate_variant)

    entity_dense_variant = _make_entity_dense_correct_variant(seed)
    if entity_dense_variant is not None:
        variants.append(entity_dense_variant)

    return _dedupe_ordered(variants)


def _build_hallucinated_response_variants(seed: SeedFact) -> tuple[str, ...]:
    variants = [
        seed.negative_response,
        f"A concise answer is: {seed.negative_response}",
    ]
    if seed.is_long_response:
        variants.append(f"In a confident summary, {seed.negative_response}")
    return _dedupe_ordered(variants)


def _is_too_trivial_or_unrealistic(seed: SeedFact) -> bool:
    negative = seed.negative_response
    positive = seed.positive_response
    if "TBD" in negative or "XXX" in negative:
        return True
    if len(negative.split()) < 4:
        return True
    return SequenceMatcher(a=positive, b=negative).ratio() > 0.985


def _make_approximate_or_range_variant(response: str) -> str | None:
    match = re.search(r"\b\d{1,4}(?:\.\d+)?\b", response)
    if match is None:
        return None
    number_text = match.group(0)
    if re.fullmatch(r"\d{4}", number_text):
        replacement = f"around {number_text}"
    else:
        replacement = f"approximately {number_text}"
    return response[: match.start()] + replacement + response[match.end() :]


def _make_entity_dense_correct_variant(seed: SeedFact) -> str | None:
    if seed.corruption_type in {"entity_swap", "place_swap", "organization_or_title_swap"}:
        return f"Historically and factually, {seed.positive_response}"
    if seed.is_long_response:
        return f"For a fact-focused overview, {seed.positive_response}"
    return None


def _dedupe_ordered(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def _bucket_flags(response: str) -> dict[str, bool]:
    token_count = len(re.findall(r"\w+", response))
    capitalized_tokens = re.findall(r"\b[A-Z][a-zA-Z-]+\b", response)
    place_match = re.search(
        r"\b(?:in|at|from|to|near|around)\s+[A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)*\b",
        response,
    )
    approximate_match = re.search(
        r"\b(?:about|around|approximately|roughly|between|from)\b",
        response.lower(),
    )
    return {
        "numbers": bool(re.search(r"\d", response)),
        "entity_like_tokens": len(capitalized_tokens) >= 2,
        "places": place_match is not None,
        "long_responses": token_count > 18,
        "approximate_or_range_style": approximate_match is not None,
    }


def _duplicate_diagnostics(
    examples: list[RawLabeledExample],
) -> dict[str, int]:
    exact_pairs = set()
    duplicate_count = 0
    near_duplicate_count = 0
    grouped_by_prompt_and_label: dict[tuple[str, int], list[str]] = {}
    for example in examples:
        pair = (example.prompt.strip(), example.response.strip(), example.label)
        if pair in exact_pairs:
            duplicate_count += 1
        else:
            exact_pairs.add(pair)
        grouped_by_prompt_and_label.setdefault(
            (example.prompt.strip(), example.label),
            [],
        ).append(example.response.strip())

    for responses in grouped_by_prompt_and_label.values():
        for index, left in enumerate(responses):
            for right in responses[index + 1 :]:
                if SequenceMatcher(a=left, b=right).ratio() > 0.985:
                    near_duplicate_count += 1
    return {
        "duplicate_count": duplicate_count,
        "near_duplicate_count": near_duplicate_count,
    }


def _dataset_warnings(
    *,
    non_hallucination_count: int,
    hallucination_count: int,
    risky_bucket_negative_coverage: dict[str, int],
    risky_bucket_positive_coverage: dict[str, int],
    too_trivial_or_unrealistic_count: int,
    sample_size: int,
) -> list[str]:
    warnings: list[str] = []
    hallucination_ratio = hallucination_count / max(sample_size, 1)
    if hallucination_ratio >= 0.48:
        warnings.append("hallucination ratio is high and may inflate false positives")
    for bucket_name, negative_coverage in risky_bucket_negative_coverage.items():
        if negative_coverage <= risky_bucket_positive_coverage[bucket_name]:
            warnings.append(
                f"correct coverage for {bucket_name} is not stronger than hallucinated coverage"
            )
    if too_trivial_or_unrealistic_count / max(sample_size, 1) > 0.05:
        warnings.append("too many examples are flagged as trivial or unrealistic")
    if non_hallucination_count <= hallucination_count:
        warnings.append("non-hallucination examples should dominate to reduce false positives")
    return warnings


def _compute_leakage_checks(
    *,
    supervision_examples: list[RawLabeledExample],
    public_eval_examples: list[RawLabeledExample],
) -> dict[str, int]:
    supervision_pairs = {
        (example.prompt.strip(), example.response.strip())
        for example in supervision_examples
    }
    supervision_prompts = {example.prompt.strip() for example in supervision_examples}

    public_pairs = {
        (example.prompt.strip(), example.response.strip())
        for example in public_eval_examples
    }
    public_prompts = {example.prompt.strip() for example in public_eval_examples}
    return {
        "public_exact_example_overlap_count": len(supervision_pairs & public_pairs),
        "public_prompt_overlap_count": len(supervision_prompts & public_prompts),
    }
