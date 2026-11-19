"""Two-passage bridge QA: SmolLM reads the first-hop passage, Qwen reads the second passage and question.

`question_only` is the second-passage-only baseline, so it measures how much the first hop adds. The
`passage1_needed` subset keeps cases where removing passage 1 changes native Qwen's answer entities,
the cases a translated first hop can help. `question_only` scores no entity matches there by construction.
"""
from data import bridge_cases

CONDITIONS = ['question_only', 'ridge', 'distilled']
SUBSETS = {'passage1_needed': lambda row: not row['conditions']['question_only']['metrics']['entities']}


def cases():
    return bridge_cases()
