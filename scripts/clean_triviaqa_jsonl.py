from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data.triviaqa_generation import clean_invalid_jsonl_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    summary = clean_invalid_jsonl_rows(
        input_path=args.input_path,
        output_path=args.output_path,
    )

    print(f"kept_row_count={summary['kept_row_count']}")
    print(f"dropped_row_count={summary['dropped_row_count']}")
    print(f"output_path={summary['output_path']}")


if __name__ == "__main__":
    main()
