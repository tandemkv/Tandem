"""Bridge QA ceiling: Qwen's own prefix cache, exact and rounded to bfloat16, beside the best map (output3).

Exact native K/V should agree with native up to judge and entity error. The bfloat16 rounding is a tiny
perturbation that shows how fragile greedy answers are. The cases and `passage1_needed` subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'native', 'native_bf16', 'output3']
