from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from models.head import LinearScoringHead


def test_predict_proba_accepts_extracted_features() -> None:
    head = LinearScoringHead()
    features = {
        "response_length": 42.0,
        "token_count_proxy": 8.0,
        "digit_count": 2.0,
        "punctuation_count": 1.0,
        "sentence_count_proxy": 1.0,
        "prompt_response_overlap": 0.25,
        "novelty_ratio_proxy": 0.75,
    }

    probability = head.predict_proba(features)

    assert isinstance(probability, float)


def test_predict_proba_returns_bounded_probability() -> None:
    head = LinearScoringHead()
    features = {
        "response_length": 5000.0,
        "token_count_proxy": 1000.0,
        "digit_count": 25.0,
        "punctuation_count": 40.0,
        "sentence_count_proxy": 35.0,
        "prompt_response_overlap": 0.0,
        "novelty_ratio_proxy": 1.0,
    }

    probability = head.predict_proba(features)

    assert 0.0 <= probability <= 1.0


def test_predict_proba_is_deterministic_for_identical_features() -> None:
    head = LinearScoringHead()
    features = {
        "response_length": 120.0,
        "token_count_proxy": 21.0,
        "digit_count": 4.0,
        "punctuation_count": 3.0,
        "sentence_count_proxy": 2.0,
        "prompt_response_overlap": 0.4,
        "novelty_ratio_proxy": 0.6,
    }

    first = head.predict_proba(features)
    second = head.predict_proba(features)

    assert first == second
