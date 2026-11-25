"""Bridge QA with the end-to-end refit map (train_output.py) against the key-refit map it started from.

The cases, order, and `passage1_needed` subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'keyfit', 'output']
