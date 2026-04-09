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
  echo "Python was not found. Install python3 or set PYTHON_BIN before running install.sh." >&2
  exit 1
fi

mkdir -p data/bench
mkdir -p data/processed
mkdir -p model/frozen_best
mkdir -p notebooks
mkdir -p reports

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e .

echo "Install complete."
echo "Manual step: place the private benchmark at data/bench/knowledge_bench_private.csv before scoring."
echo "Manual step: ensure configs/token_stat_provider.local.json points to the local GigaChat checkpoint before training or scoring."
echo "Kaggle note: if you run inside Kaggle, prefer configs/token_stat_provider.kaggle.json and keep the repo under /kaggle/working."
echo "Kaggle note: reports will be written to /kaggle/working/<repo>/reports when you use the default paths."
