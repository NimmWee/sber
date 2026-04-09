#!/usr/bin/env bash
set -euo pipefail

python scripts/score_frozen_submission.py "$@"
