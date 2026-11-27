"""Bridge QA with the first prefix tokens computed natively by Qwen (attention sinks) on top of output2.

The cases, order, and `passage1_needed` subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output2', 'header1_output2', 'header5_output2']
