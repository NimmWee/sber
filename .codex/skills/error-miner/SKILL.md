---
name: error-miner
description: Analyze failure modes and propose targeted fixes.
---

# Goal
Turn false positives and false negatives into ranked next experiments.

When invoked:
1. Group false positives and false negatives by taxonomy:
   - dates,
   - names,
   - places,
   - titles/relations,
   - numbers,
   - long-tail facts.
2. Identify which feature groups fail.
3. Propose 3-5 concrete next experiments ranked by expected gain vs cost.
