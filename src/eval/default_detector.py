from collections.abc import Mapping

from eval.ablation import build_feature_allowlist, filter_feature_rows
from features.extractor import StructuralFeatureExtractor
from models.head import train_lightgbm_head


DEFAULT_TOKEN_FEATURE_GROUPS = ("base_token_uncertainty",)
DEFAULT_HEAD_KIND = "lightgbm"


def build_default_detector_extractor() -> StructuralFeatureExtractor:
    return StructuralFeatureExtractor(
        enable_token_uncertainty=True,
        enable_internal_features=True,
        enable_compact_internal_enhancements=True,
        token_feature_groups=DEFAULT_TOKEN_FEATURE_GROUPS,
    )


def build_default_detector_feature_allowlist(
    *,
    feature_rows: list[Mapping[str, float]],
) -> set[str]:
    structural_feature_names = {
        feature_name
        for feature_row in feature_rows
        for feature_name in feature_row
        if not feature_name.startswith("token_")
        and not feature_name.startswith("internal_")
    }
    allowlist = build_feature_allowlist(
        structural_feature_names=structural_feature_names,
        enabled_groups=DEFAULT_TOKEN_FEATURE_GROUPS,
    )
    internal_feature_names = {
        feature_name
        for feature_row in feature_rows
        for feature_name in feature_row
        if feature_name.startswith("internal_")
    }
    return allowlist | internal_feature_names


def filter_default_detector_rows(
    *,
    feature_rows: list[Mapping[str, float]],
) -> list[dict[str, float]]:
    allowlist = build_default_detector_feature_allowlist(feature_rows=feature_rows)
    return filter_feature_rows(
        feature_rows=feature_rows,
        allowed_features=allowlist,
    )


def train_default_detector_head(
    *,
    feature_rows: list[Mapping[str, float]],
    labels: list[int],
):
    filtered_rows = filter_default_detector_rows(feature_rows=feature_rows)
    return train_lightgbm_head(filtered_rows, labels)
