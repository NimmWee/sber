from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.scorer import score


def test_score_returns_float_for_prompt_and_response() -> None:
    result = score(prompt="What is the capital of France?", response="Paris.")

    assert isinstance(result, float)


def test_score_returns_probability_bounded_to_unit_interval() -> None:
    result = score(prompt="Name a prime number.", response="2")

    assert 0.0 <= result <= 1.0


def test_score_is_deterministic_for_same_input() -> None:
    prompt = "Who wrote Hamlet?"
    response = "William Shakespeare."

    first = score(prompt=prompt, response=response)
    second = score(prompt=prompt, response=response)

    assert first == second
