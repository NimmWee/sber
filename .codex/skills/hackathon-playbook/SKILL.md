---
name: hackathon-playbook
description: Orchestrate work on the Guardian of Truth project using TDD, evaluation discipline, and production-first tradeoffs.
---

# Default strategy
- Reproduce baseline ideas first
- Stabilize evaluation
- Add one cheap feature family at a time
- Use latency checks after each meaningful change
- Prefer simple ensembles over complex pipelines
- Keep research and production paths separate

# Priority order
1. Working scorer contract
2. Stable eval
3. Uncertainty features
4. Internal-state features
5. Lightweight ensemble
6. Explainability
7. Demo polish

# Never do first
- heavy RAG in runtime
- repeated generation in runtime
- untested feature dumps
- broad refactors without tests
