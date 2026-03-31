---
name: latency-guard
description: Check whether a code or model change threatens the 500 ms detector budget and reject expensive changes without strong evidence.
---

# Goal
Protect inference speed.

# Workflow
1. Identify which path changed: feature extraction, classifier, preprocessing, orchestration.
2. Run the latency benchmark.
3. Compare against baseline.
4. Attribute the regression.
5. Recommend keep, optimize, or revert.

# Rules
- Treat latency as a first-class metric.
- Expensive features must show clear quality gain.
- Do not allow hidden extra passes in production mode.

# Output
- baseline latency
- new latency
- delta
- likely cause
- recommendation
