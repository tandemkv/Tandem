"""Context-only handoff: SmolLM prefills the document, Qwen reads the question and answers."""
from data import hotpot_test

CONDITIONS = ['question_only', 'ridge', 'distilled', 'permuted_distilled']


def cases():
    return hotpot_test()
