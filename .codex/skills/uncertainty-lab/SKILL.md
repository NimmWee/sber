---
name: uncertainty-lab
description: Build fast uncertainty features from token probabilities and response structure.
---

# Goal
Engineer cheap, robust uncertainty features from logits, logprobs, and answer structure.

When invoked:
1. Extract token-level uncertainty features:
   - mean/min logprob,
   - entropy,
   - top1-top2 margin,
   - tail low-confidence rate,
   - confidence decay over response positions.
2. Add response-structure features:
   - length,
   - numbers/dates/entities count,
   - prompt-response overlap,
   - novelty of answer spans.
3. Prefer features that are cheap and robust.
4. Output latency-cost estimates for each feature group.
