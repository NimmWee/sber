from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eval.ablation import (
    FEATURE_GROUP_ORDER,
    TOKEN_BASE_FEATURES,
    TOKEN_GROUP_FEATURES,
    build_feature_allowlist,
    filter_feature_rows,
    recommend_feature_group,
)
from eval.metrics import compute_pr_auc
from features.extractor import StructuralFeatureExtractor
from inference.token_stats import TransformersTokenStatProvider
from models.head import train_logistic_regression_head
from utils.script_helpers import (
    build_ablation_examples,
    resolve_transformers_provider_config,
    write_json_artifact,
)


GROUP_HELPERS = {
    "base_token_uncertainty": StructuralFeatureExtractor._extract_token_uncertainty_base,
    "variance_std": StructuralFeatureExtractor._extract_token_uncertainty_variance_std,
    "segment_summaries": StructuralFeatureExtractor._extract_token_uncertainty_segment_summaries,
    "span_tail_rates": StructuralFeatureExtractor._extract_token_uncertainty_span_tail_rates,
    "specialized_tokens": StructuralFeatureExtractor._extract_token_uncertainty_specialized,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "ablation_real_provider"),
    )
    parser.add_argument("--latency-repeat-count", type=int, default=50)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    extractor = StructuralFeatureExtractor(enable_token_uncertainty=True)
    train_examples, validation_examples = build_ablation_examples()

    train_rows, train_labels, train_token_stats = _prepare_rows(
        examples=train_examples,
        provider=provider,
        extractor=extractor,
    )
    validation_rows, validation_labels, validation_token_stats = _prepare_rows(
        examples=validation_examples,
        provider=provider,
        extractor=extractor,
    )

    all_feature_names = set(train_rows[0].keys())
    token_feature_names = set(TOKEN_BASE_FEATURES)
    for feature_names in TOKEN_GROUP_FEATURES.values():
        token_feature_names.update(feature_names)
    structural_feature_names = all_feature_names - token_feature_names

    baseline_allowlist = build_feature_allowlist(
        structural_feature_names=structural_feature_names,
        enabled_groups=("base_token_uncertainty",),
    )
    baseline_pr_auc = _score_with_allowlist(
        train_rows=train_rows,
        train_labels=train_labels,
        validation_rows=validation_rows,
        validation_labels=validation_labels,
        allowed_features=baseline_allowlist,
    )

    baseline_latency = _measure_group_latency(
        token_stats_list=train_token_stats + validation_token_stats,
        enabled_groups=("base_token_uncertainty",),
        repeat_count=args.latency_repeat_count,
    )

    comparison_rows = []
    for group_name in FEATURE_GROUP_ORDER[1:]:
        allowlist = build_feature_allowlist(
            structural_feature_names=structural_feature_names,
            enabled_groups=("base_token_uncertainty", group_name),
        )
        pr_auc = _score_with_allowlist(
            train_rows=train_rows,
            train_labels=train_labels,
            validation_rows=validation_rows,
            validation_labels=validation_labels,
            allowed_features=allowlist,
        )
        aggregation_latency_ms = _measure_group_latency(
            token_stats_list=train_token_stats + validation_token_stats,
            enabled_groups=("base_token_uncertainty", group_name),
            repeat_count=args.latency_repeat_count,
        )
        comparison_rows.append(
            {
                "group": group_name,
                "pr_auc": pr_auc,
                "pr_auc_delta_vs_base": pr_auc - baseline_pr_auc,
                "aggregation_latency_mean_ms": aggregation_latency_ms,
                "aggregation_latency_delta_ms": aggregation_latency_ms - baseline_latency,
                "recommendation": recommend_feature_group(
                    pr_auc_delta=pr_auc - baseline_pr_auc,
                    latency_delta_ms=aggregation_latency_ms - baseline_latency,
                ),
            }
        )

    full_allowlist = build_feature_allowlist(
        structural_feature_names=structural_feature_names,
        enabled_groups=FEATURE_GROUP_ORDER,
    )
    full_pr_auc = _score_with_allowlist(
        train_rows=train_rows,
        train_labels=train_labels,
        validation_rows=validation_rows,
        validation_labels=validation_labels,
        allowed_features=full_allowlist,
    )
    full_latency = _measure_group_latency(
        token_stats_list=train_token_stats + validation_token_stats,
        enabled_groups=FEATURE_GROUP_ORDER,
        repeat_count=args.latency_repeat_count,
    )

    payload = {
        "model_source": config.model_source,
        "response_delimiter": config.response_delimiter,
        "sample_sizes": {
            "train": len(train_examples),
            "validation": len(validation_examples),
        },
        "baseline": {
            "group": "base_token_uncertainty",
            "pr_auc": baseline_pr_auc,
            "aggregation_latency_mean_ms": baseline_latency,
        },
        "groups": comparison_rows,
        "full_enriched": {
            "pr_auc": full_pr_auc,
            "pr_auc_delta_vs_base": full_pr_auc - baseline_pr_auc,
            "aggregation_latency_mean_ms": full_latency,
            "aggregation_latency_delta_ms": full_latency - baseline_latency,
        },
    }
    artifact_path = write_json_artifact(
        artifact_dir=artifact_dir,
        filename="ablation_real_provider_summary.json",
        payload=payload,
    )

    print(f"model={config.model_source}")
    print(f"base_pr_auc={baseline_pr_auc:.4f}")
    for row in comparison_rows:
        print(
            f"{row['group']} pr_auc={row['pr_auc']:.4f} "
            f"delta={row['pr_auc_delta_vs_base']:.4f} "
            f"latency_delta_ms={row['aggregation_latency_delta_ms']:.4f} "
            f"recommendation={row['recommendation']}"
        )
    print(
        f"full_enriched_pr_auc={full_pr_auc:.4f} "
        f"delta={full_pr_auc - baseline_pr_auc:.4f}"
    )
    print(f"artifact={artifact_path}")


def _prepare_rows(*, examples, provider, extractor):
    rows = []
    labels = []
    token_stats_list = []
    for example in examples:
        token_stats = provider.collect(prompt=example.prompt, response=example.response)
        token_stats_list.append(token_stats)
        rows.append(
            dict(
                extractor.extract(
                    prompt=example.prompt,
                    response=example.response,
                    token_stats=token_stats,
                )
            )
        )
        labels.append(example.label)
    return rows, labels, token_stats_list


def _score_with_allowlist(
    *,
    train_rows,
    train_labels,
    validation_rows,
    validation_labels,
    allowed_features,
) -> float:
    filtered_train = filter_feature_rows(
        feature_rows=train_rows,
        allowed_features=allowed_features,
    )
    filtered_validation = filter_feature_rows(
        feature_rows=validation_rows,
        allowed_features=allowed_features,
    )
    model = train_logistic_regression_head(filtered_train, train_labels)
    probabilities = model.predict_proba_batch(filtered_validation)
    return compute_pr_auc(validation_labels, probabilities)


def _measure_group_latency(*, token_stats_list, enabled_groups, repeat_count: int) -> float:
    helper_sequence = [GROUP_HELPERS[group_name] for group_name in enabled_groups]
    samples_ms = []
    for _ in range(repeat_count):
        start = time.perf_counter()
        for token_stats in token_stats_list:
            for helper in helper_sequence:
                helper(token_stats)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return statistics.fmean(samples_ms)


if __name__ == "__main__":
    main()
