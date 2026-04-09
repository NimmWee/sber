#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/bench
mkdir -p data/processed
mkdir -p model/frozen_best
mkdir -p notebooks

python -m pip install --upgrade pip
python -m pip install -e .

echo "Install complete."
echo "Manual step: place the private benchmark at data/bench/knowledge_bench_private.csv before scoring."
echo "Manual step: ensure configs/token_stat_provider.local.json points to the local GigaChat checkpoint before training or scoring."
