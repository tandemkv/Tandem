"""Evaluation cases. A case is one prompt whose document prefix ends at a question boundary."""
import random
from dataclasses import dataclass

from harness import test_rows


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str  # Document, then '\n\nQuestion: ...'. Only the document is translated.
    question: str
    gold: str | None = None  # Optional. The primary reference is native Qwen.


def hotpot_test(seed=20261113):
    """The 240 pinned held-out HotpotQA prompts in a fixed shuffled order."""
    rows = [row for row in test_rows() if row['dataset'] == 'hotpot']
    random.Random(seed).shuffle(rows)
    return [Case(row['id'], row['prompt'], row['question'], row['answer']) for row in rows]
