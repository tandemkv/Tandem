"""Bridge QA: does training with native header tokens (output3h5) fix what adding them at test time broke?

output3 is the control with the same extra steps and no header. The cases, order, and `passage1_needed`
subset match `bridge`, so `--count 500` repeats its run.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output3', 'header5_output3', 'header5_output3h5']
