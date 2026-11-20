"""Two-passage bridge QA: SmolLM reads the first-hop passage, Qwen reads the second passage and question.

`question_only` is the second-passage-only baseline, so it measures how much the first hop adds. The
`passage1_needed` subset keeps cases where removing passage 1 changes native Qwen's answer, the cases a
translated first hop can help. It uses the judge verdict when present, else the entity match.
"""
from data import bridge_cases

CONDITIONS = ['question_only', 'ridge', 'distilled']


def passage1_needed(row):
    metrics = row['conditions']['question_only']['metrics']
    return not metrics.get('judge', metrics['entities'])


SUBSETS = {'passage1_needed': passage1_needed}


def cases():
    return bridge_cases()
