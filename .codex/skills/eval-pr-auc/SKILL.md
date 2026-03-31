---
name: eval-pr-auc
description: Run the standard evaluation workflow for the hackathon detector, including PR-AUC, calibration checks, and error slicing.
---

# Goal
Produce a consistent evaluation report.

# Workflow
1. Verify dataset split and no leakage assumptions.
2. Run scorer on the target split.
3. Compute PR-AUC.
4. Compute threshold-free diagnostics and optional calibration metrics.
5. Slice errors by response length, named entities, dates, numbers, and confidence buckets.
6. Save artifacts and summarize findings.

# Rules
- PR-AUC is the primary metric.
- Always note sample size and split.
- Keep reports comparable between runs.

# Output
- dataset/split
- model/version
- PR-AUC
- slices
- notable regressions
- recommendation
