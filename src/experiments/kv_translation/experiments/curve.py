"""Bridge QA on output3: agreement against the share of passage-1 tokens that Qwen computes itself.

Scattered random tokens, the passage lead, and the contiguous window SmolLM's last prefix token attends to
most (hub), at 15, 33 and 50%, refreshed over the translated cache. The scattered copy oracle runs from 15
to 75% to show how agreement scales. Cases and the `passage1_needed` subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = (['question_only', 'output3']
              + [f'oracle_scattered{p}_output3' for p in (15, 33, 50, 75)]
              + [f'refresh_{name}{p}_output3' for name in ('scattered', 'lead', 'hub') for p in (15, 33, 50)])
