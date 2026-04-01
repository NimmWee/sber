from __future__ import annotations

from dataclasses import dataclass

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
        prompt="What is the capital of Australia?",
        positive_response="Canberra is the capital city of Australia.",
        negative_response="Sydney is the capital city of Australia.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="Who wrote The Odyssey?",
        positive_response="Homer wrote The Odyssey.",
        negative_response="Virgil wrote The Odyssey.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="How many planets are in the Solar System?",
        positive_response="There are 8 planets in the Solar System.",
        negative_response="There are 9 planets in the Solar System.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="When did Apollo 11 land on the Moon?",
        positive_response="Apollo 11 landed on the Moon in 1969.",
        negative_response="Apollo 11 landed on the Moon in 1971.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="Which company developed the Windows operating system?",
        positive_response="Microsoft developed the Windows operating system.",
        negative_response="IBM developed the Windows operating system.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Who painted Guernica?",
        positive_response="Pablo Picasso painted Guernica.",
        negative_response="Salvador Dali painted Guernica.",
        corruption_type="entity_swap",
    ),
    SeedFact(
        prompt="What is the tallest mountain in Japan?",
        positive_response="Mount Fuji is the tallest mountain in Japan.",
        negative_response="Mount Aso is the tallest mountain in Japan.",
        corruption_type="place_swap",
    ),
    SeedFact(
        prompt="How many chambers are in the human heart?",
        positive_response="The human heart has 4 chambers.",
        negative_response="The human heart has 5 chambers.",
        corruption_type="number_nearby",
    ),
    SeedFact(
        prompt="When was the Berlin Wall opened?",
        positive_response="The Berlin Wall was opened in 1989.",
        negative_response="The Berlin Wall was opened in 1991.",
        corruption_type="date_nearby",
    ),
    SeedFact(
        prompt="Which organization operates the Hubble Space Telescope?",
        positive_response="NASA operates the Hubble Space Telescope.",
        negative_response="ESA operates the Hubble Space Telescope.",
        corruption_type="organization_or_title_swap",
    ),
    SeedFact(
        prompt="Summarize the Voyager 1 mission milestone.",
        positive_response=(
            "Voyager 1 was launched in 1977 by NASA and became the first spacecraft "
            "to enter interstellar space after completing extended observations of Jupiter and Saturn."
        ),
        negative_response=(
            "Voyager 1 was launched in 1981 by NASA and became the first spacecraft "
            "to enter interstellar space after completing extended observations of Jupiter and Saturn."
        ),
        corruption_type="date_nearby",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Summarize the role of Alexander Fleming in the history of medicine.",
        positive_response=(
            "Alexander Fleming discovered penicillin in 1928, and the finding later transformed "
            "modern medicine by enabling widely effective antibiotic treatment."
        ),
        negative_response=(
            "Alexander Fleming discovered penicillin in 1938, and the finding later transformed "
            "modern medicine by enabling widely effective antibiotic treatment."
        ),
        corruption_type="date_nearby",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Explain who led the first successful ascent of Mount Everest.",
        positive_response=(
            "Edmund Hillary and Tenzing Norgay led the first successful ascent of Mount Everest in 1953, "
            "a milestone expedition organized with British support."
        ),
        negative_response=(
            "George Mallory and Tenzing Norgay led the first successful ascent of Mount Everest in 1953, "
            "a milestone expedition organized with British support."
        ),
        corruption_type="entity_swap",
        is_long_response=True,
    ),
    SeedFact(
        prompt="Describe which city hosts the headquarters of the European Parliament.",
        positive_response=(
            "The European Parliament maintains its official seat in Strasbourg, while other parliamentary work "
            "also takes place in Brussels and Luxembourg."
        ),
        negative_response=(
            "The European Parliament maintains its official seat in Brussels, while other parliamentary work "
            "also takes place in Brussels and Luxembourg."
        ),
        corruption_type="place_swap",
        is_long_response=True,
    ),
)


def build_non_public_supervision_dataset(
    *,
    public_eval_examples: list[RawLabeledExample] | None = None,
) -> NonPublicSupervisionDataset:
    examples: list[tuple[RawLabeledExample, str, bool]] = []
    for seed in SEED_FACTS:
        examples.append(
            (
                RawLabeledExample(
                    prompt=seed.prompt,
                    response=seed.positive_response,
                    label=0,
                ),
                "clean_positive",
                seed.is_long_response,
            )
        )
        examples.append(
            (
                RawLabeledExample(
                    prompt=seed.prompt,
                    response=seed.negative_response,
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

    for index, (example, corruption_type, is_long_response) in enumerate(examples):
        if example.label == 0:
            positive_count += 1
        else:
            negative_count += 1
            corruption_taxonomy[corruption_type] += 1
        if is_long_response:
            long_response_count += 1

        if index % 5 == 0:
            dev_examples.append(example)
        else:
            train_examples.append(example)

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
        "label_balance": {
            "positive_ratio": positive_count / (positive_count + negative_count),
            "negative_ratio": negative_count / (positive_count + negative_count),
        },
        "long_response_count": long_response_count,
        "corruption_taxonomy": corruption_taxonomy,
        "leakage_checks": leakage_checks,
    }
    return NonPublicSupervisionDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        summary=summary,
    )


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
