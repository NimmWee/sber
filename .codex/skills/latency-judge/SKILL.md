---
name: latency-judge
description: Measure, guard, and optimize latency for single-example scoring.
---

# Goal
Measure and protect single-example scoring latency.

When invoked:
1. Benchmark:
   - feature extraction,
   - head inference,
   - full score() path.
2. Report p50/p95 and worst offenders.
3. Suggest:
   - vectorization,
   - caching,
   - layer reduction,
   - gray-zone escalation rules.
4. Reject changes that threaten the 500 ms budget without quality justification.
