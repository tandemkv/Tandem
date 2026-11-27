"""Bridge QA with the continued end-to-end refit (output2) against the 4000-step refit it continued.

The cases, order, and `passage1_needed` subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output', 'output2']
