#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/bench
mkdir -p data/processed
mkdir -p data/textual
mkdir -p model/frozen_best
mkdir -p notebooks

python -m pip install --upgrade pip
python -m pip install -e .
