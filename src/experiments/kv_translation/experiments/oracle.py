"""Bridge QA oracle on output3: native keys or values, early or late layers, to locate the remaining loss.

The cases, order, and `passage1_needed` subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output3', 'native_keys_output3', 'native_values_output3',
              'native_early_output3', 'native_late_output3']
