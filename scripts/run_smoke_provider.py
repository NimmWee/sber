from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from inference.token_stats import TransformersTokenStatProvider
from utils.script_helpers import (
    build_smoke_examples,
    resolve_transformers_provider_config,
    write_json_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "smoke_provider"),
    )
    args = parser.parse_args()

    config = resolve_transformers_provider_config(
        project_root=PROJECT_ROOT,
        explicit_config_path=args.config,
    )
    provider = TransformersTokenStatProvider(config=config)
    _, validation_examples = build_smoke_examples()

    smoke_rows = []
    for example in validation_examples:
        token_stats = provider.collect(
            prompt=example.prompt,
            response=example.response,
        )
        smoke_rows.append(
            {
                "prompt": example.prompt,
                "response": example.response,
                "token_count": len(token_stats),
                "tokens": [ascii(stat.token) for stat in token_stats[:5]],
            }
        )

    payload = {
        "model_source": config.model_source,
        "response_delimiter": config.response_delimiter,
        "rows": smoke_rows,
    }
    artifact_path = write_json_artifact(
        artifact_dir=args.artifact_dir,
        filename="smoke_provider_summary.json",
        payload=payload,
    )

    print(f"model={config.model_source}")
    print(f"delimiter={config.response_delimiter!r}")
    print(f"examples={len(smoke_rows)}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
