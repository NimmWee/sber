---
name: feature-research
description: Evaluate one hallucination-detection feature hypothesis at a time with explicit quality and latency tradeoff analysis.
---

# Goal
Turn a feature idea into a small, measurable experiment.

# Workflow
1. State the hypothesis.
2. Define the exact signals needed.
3. Estimate runtime cost.
4. Add tests for extractor contracts.
5. Implement the smallest viable extractor.
6. Run eval and latency checks.
7. Decide: keep, drop, or iterate.

# Rules
- Only one feature family per experiment step.
- No hidden benchmark leakage.
- No runtime external APIs.
- Prefer cheap deterministic features.

# Required report
- hypothesis
- extractor contract
- eval result
- latency impact
- keep/drop decision
