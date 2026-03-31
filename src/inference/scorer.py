from features.extractor import StructuralFeatureExtractor
from models.head import LinearScoringHead


DEFAULT_EXTRACTOR = StructuralFeatureExtractor()
DEFAULT_HEAD = LinearScoringHead()


def score(prompt: str, response: str) -> float:
    features = DEFAULT_EXTRACTOR.extract(prompt=prompt, response=response)
    return DEFAULT_HEAD.predict_proba(features)
