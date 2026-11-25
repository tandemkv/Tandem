"""Bridge QA with the key-refit map (train_keys.py) against the distilled map it started from.

The cases, order, and `passage1_needed` subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'distilled', 'keyfit']
