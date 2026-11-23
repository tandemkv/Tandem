"""Evaluation cases. A case is one prompt split into a translated prefix and a native suffix."""
import hashlib
import random
import unicodedata
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from harness import split_rows

HOTPOT = ('hotpotqa/hotpot_qa', '1908d6afbbead072334abe2965f91bd2709910ab',
          'distractor/validation-00000-of-00001.parquet')


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    question: str
    gold: str | None = None  # Optional. The primary reference is native Qwen.
    split_at: str = '\n\nQuestion: '  # Text before the last occurrence is translated; the rest is native.


def hotpot_test(seed=20261113):
    """The 240 pinned held-out HotpotQA prompts in a fixed shuffled order."""
    rows = [row for row in split_rows('test') if row['dataset'] == 'hotpot']
    random.Random(seed).shuffle(rows)
    return [Case(row['id'], row['prompt'], row['question'], row['answer']) for row in rows]


def document_key(title):
    """The document hash the pinned splits use to keep training and test documents apart."""
    return hashlib.sha256(unicodedata.normalize('NFKC', title).casefold().strip().encode()).hexdigest()


def bridge_cases(seed=20261119):
    """HotpotQA validation bridge questions: the first-hop passage is translated, the second is native.

    Passage 2 is the supporting passage that contains the gold answer and Passage 1 is the other one.
    Questions whose answer is yes/no, missing from both passages, or present in both are skipped, as are
    questions that share an ID or document with the pinned training or test splits.
    """
    pinned = split_rows('train') + split_rows('test')
    seen_ids = {row['id'] for row in pinned}
    seen_docs = {doc for row in pinned for doc in row.get('documents', [])}
    table = pq.read_table(hf_hub_download(HOTPOT[0], HOTPOT[2], repo_type='dataset', revision=HOTPOT[1]))
    cases = []
    for row in table.to_pylist():
        titles, bodies = row['context']['title'], row['context']['sentences']
        support = list(dict.fromkeys(row['supporting_facts']['title']))
        answer = row['answer'].strip()
        if (row['type'] != 'bridge' or row['id'] in seen_ids or len(support) != 2
                or not set(support) <= set(titles) or answer.lower() in ('yes', 'no')
                or {document_key(t) for t in titles} & seen_docs):
            continue
        passages = {t: f"{t}\n{''.join(bodies[titles.index(t)]).strip()}" for t in support}
        holds = [t for t in support if answer in passages[t]]
        if len(holds) != 1:
            continue
        second = holds[0]
        first = next(t for t in support if t != second)
        prompt = (f'Passage 1:\n{passages[first]}\n\nPassage 2:\n{passages[second]}'
                  f'\n\nQuestion: {row["question"].strip()}\nAnswer:')
        cases.append(Case(row['id'], prompt, row['question'], answer, split_at='\n\nPassage 2:'))
    random.Random(seed).shuffle(cases)
    return cases
