"""Two-passage bridge QA: SmolLM reads the first-hop passage, Qwen reads the second passage and question.

`question_only` is the second-passage-only baseline, so it measures how much the first hop adds.
"""
from data import bridge_cases

CONDITIONS = ['question_only', 'ridge', 'distilled']


def cases():
    return bridge_cases()
