---
name: tdd-implement
description: Use strict TDD for implementing or changing code. Trigger for any feature, bugfix, refactor with behavior change, or API contract work.
---

# Goal
Implement changes using strict Red -> Green -> Refactor.

# Workflow
1. Restate the smallest next behavior.
2. Write exactly one failing test first.
3. Run the narrowest relevant test and confirm failure.
4. Implement the minimum code needed to pass.
5. Refactor safely while keeping tests green.
6. Run broader relevant tests.
7. Summarize what changed and any remaining risk.

# Rules
- Do not implement ahead of tests.
- Do not batch multiple behaviors into one test.
- Prefer deterministic tests over brittle mocks.
- If the code is hard to test, improve the design.
- For bugfixes, always start with a regression test.

# Output format
- behavior
- failing test added
- code change
- tests run
- result
- next smallest step
