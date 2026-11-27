"""Bridge QA with output4 (output3 trained through the rest of a second pass) against output3.

The cases, order, and `passage1_needed` subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output3', 'output4']
