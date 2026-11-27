"""Bridge QA on output3: are translated tokens harmful, or only lossy, once some native columns exist?

Copy oracle with the other body tokens translated (oracle_scattered) against the same cache with them masked
out of attention (masked_scattered), at 0, 15, 33, 50 and 75% native. Cases and the subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = (['question_only', 'output3', 'masked_scattered0_output3']
              + [f'{kind}_scattered{p}_output3' for p in (15, 33, 50, 75) for kind in ('oracle', 'masked')])
