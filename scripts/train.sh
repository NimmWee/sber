#!/usr/bin/env bash
set -euo pipefail

python scripts/build_text_training_dataset.py
python scripts/train_text_detector.py "$@"
