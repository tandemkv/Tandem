# Agent instructions

## Python tooling

Prefer `uv` for Python environments, dependencies, tools, and scripts. Use another tool only when `uv` is unavailable or unsuitable.

## Minimal pull requests

Every PR must contain the smallest complete change needed for its stated purpose. Keep each PR focused on one task and split independent work into separate PRs. Reuse existing code and conventions. Avoid unrelated cleanup, speculative abstractions, unnecessary dependencies, duplicate documentation, and generated data or reports unless required by the task. Keep tests and documentation proportional to the change without dropping necessary correctness checks or reproducibility files. Before opening a PR, review the entire diff and remove anything that does not directly support the task.
