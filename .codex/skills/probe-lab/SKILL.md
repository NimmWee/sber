---
name: probe-lab
description: Engineer internal-model-signal features from hidden states, activations, and intermediate layers.
---

# Goal
Engineer hidden-state, activation, and layer-disagreement features.

When invoked:
1. Add or refine hidden-state feature extraction.
2. Prefer a small selected set of layers before full-layer extraction.
3. Compute:
   - pooled answer embeddings,
   - layerwise norms/variance,
   - layer disagreement,
   - answer-only and entity-aware pooling,
   - intermediate logit-derived summaries if cheap.
4. Watch memory and device placement carefully.
5. Produce a compact feature table and timing breakdown.
