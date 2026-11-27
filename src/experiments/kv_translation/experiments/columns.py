"""Bridge QA token-column oracles and selective refresh on output3.

Oracles copy native Qwen K/V (every layer) into the selected prefix tokens; refresh recomputes them with Qwen
over the translated cache of the rest, which needs no native prefix. Selections: fractured tokenizer spans
(about 7% of prefix tokens), entity words (about 37%), words Qwen splits into pieces, and random sets of matched size. Cases and the
`passage1_needed` subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output3',
              'oracle_fractured_output3', 'oracle_random_fractured_output3',
              'oracle_entity_output3', 'oracle_random_entity_output3',
              'refresh_fractured_output3', 'refresh_random_fractured_output3',
              'refresh_entity_output3', 'refresh_random_entity_output3',
              'oracle_multi_output3', 'oracle_random_multi_output3',
              'refresh_multi_output3', 'refresh_random_multi_output3']
