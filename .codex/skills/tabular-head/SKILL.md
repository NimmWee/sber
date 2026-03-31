---
name: tabular-head
description: Train, compare, and calibrate lightweight heads for hallucination scoring.
---

# Goal
Train and compare lightweight tabular heads under quality and latency constraints.

When invoked:
1. Compare Logistic Regression, LightGBM, and CatBoost.
2. Optimize for PR-AUC under latency constraints.
3. Always return:
   - validation PR-AUC,
   - calibration quality,
   - feature importance or coefficients,
   - inference latency.
4. Prefer the simplest model that wins.
