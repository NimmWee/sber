# AGENTS.md

## Mission

Build a production-grade factual hallucination detector for the sber hackathon.

The system must:
- Take as input: `prompt + response`
- Output: `hallucination_probability ∈ [0, 1]`
- Optimize for: **PR-AUC**
- Respect strict latency: **≤ 500 ms per example**
- Avoid external APIs in the runtime scoring path
- Prefer internal model signals, uncertainty features, and lightweight classifiers

---

## Core Principles

1. **TDD is mandatory**
2. **Latency is a first-class constraint**
3. **Evaluation correctness > model complexity**
4. **Small, measurable improvements only**
5. **Reproducibility over cleverness**

---

## Working Style (STRICT TDD)

All implementation must follow:

1. Restate the smallest next behavior
2. Write a failing test (**Red**)
3. Run the narrowest relevant test
4. Implement minimal code (**Green**)
5. Refactor safely (**Refactor**)
6. Re-run tests
7. Document what changed

### Hard rules

- Never write implementation before a failing test
- Never batch multiple behaviors in one test
- Every bugfix MUST start with a regression test
- If code is hard to test → redesign it
- Keep each step small and reviewable

---

## Hackathon Constraints

### MUST follow

- Do not use preview benchmark as training data
- Do not introduce external APIs in runtime
- Do not rely on multi-pass generation in scoring path
- Do not break the `prompt + response → score` contract

### Prefer

- Uncertainty features (logprobs, entropy, margin)
- Internal model signals (hidden states, layer disagreement)
- Lightweight tabular models
- Simple ensembles with clear latency accounting

### Avoid

- Heavy RAG in production path
- Repeated generation
- Complex pipelines without measurable gain

---

## Architecture Guidelines

### Production path (`src/inference/`)

- Single forward pass
- Feature extraction (fast, deterministic)
- Lightweight scoring head
- Output probability + optional explanation

### Research path (`experiments/`)

- Feature experiments
- Ablations
- Calibration
- Error analysis

---

## Feature Engineering Rules

Preferred features:

- Token log probabilities
- Entropy / surprisal
- Top1-top2 margin
- Tail low-confidence rate
- Confidence decay over response
- Hidden state statistics
- Layer disagreement
- Response structure (length, numbers, entities)

All features must:
- Be deterministic
- Be testable
- Have known latency cost
- Avoid NaN/Inf

---

## Evaluation Rules

Primary metric: **PR-AUC**

Every evaluation must include:

- dataset / split description
- PR-AUC
- error slices:
  - dates
  - names
  - places
  - numbers
- regression comparison vs previous version

Never:
- mix train/dev/test implicitly
- leak benchmark into training

---

## Latency Rules

Latency is critical.

Always measure:

- feature extraction time
- model inference time
- full `score()` path

Track:
- p50
- p95

Reject changes if:
- latency increases significantly
- without clear PR-AUC gain

---

## Definition of Done

A task is DONE only if:

- behavior is covered by tests
- tests pass
- lint passes
- typecheck passes
- evaluation was run (if relevant)
- latency impact is known
- no external API was introduced
- changes are documented

---

## Repository Structure
