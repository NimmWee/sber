#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"

if [ -n "${PYTHON_BIN}" ]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python was not found. Install python3 or set PYTHON_BIN before running train.sh." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/build_text_training_dataset.py
"${PYTHON_BIN}" scripts/train_frozen_submission.py "$@"
