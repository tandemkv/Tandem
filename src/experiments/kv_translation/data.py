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


HOTPOT_TRAIN = 'distractor/train-00000-of-00002.parquet'


def bridge_case(row):
    """One bridge question as a Case: the first-hop passage is translated, the second is native, or None.

    Passage 2 is the supporting passage that contains the gold answer and Passage 1 is the other one.
    Questions whose answer is yes/no, missing from both passages, or present in both are skipped.
    """
    titles, bodies = row['context']['title'], row['context']['sentences']
    support = list(dict.fromkeys(row['supporting_facts']['title']))
    answer = row['answer'].strip()
    if (row['type'] != 'bridge' or len(support) != 2 or not set(support) <= set(titles)
            or answer.lower() in ('yes', 'no')):
        return None
    passages = {t: f"{t}\n{''.join(bodies[titles.index(t)]).strip()}" for t in support}
    holds = [t for t in support if answer in passages[t]]
    if len(holds) != 1:
        return None
    second = holds[0]
    first = next(t for t in support if t != second)
    prompt = (f'Passage 1:\n{passages[first]}\n\nPassage 2:\n{passages[second]}'
              f'\n\nQuestion: {row["question"].strip()}\nAnswer:')
    return Case(row['id'], prompt, row['question'], answer, split_at='\n\nPassage 2:')


def hotpot_rows(path):
    return pq.read_table(hf_hub_download(HOTPOT[0], path, repo_type='dataset', revision=HOTPOT[1])).to_pylist()


def bridge_cases(seed=20261119):
    """HotpotQA validation bridge questions (see bridge_case), skipping questions that share an ID or
    document with the pinned training or test splits."""
    pinned = split_rows('train') + split_rows('test')
    seen_ids = {row['id'] for row in pinned}
    seen_docs = {doc for row in pinned for doc in row.get('documents', [])}
    cases = [bridge_case(row) for row in hotpot_rows(HOTPOT[2])
             if row['id'] not in seen_ids and not {document_key(t) for t in row['context']['title']} & seen_docs]
    cases = [case for case in cases if case]
    random.Random(seed).shuffle(cases)
    return cases


def bridge_train_cases(seed=20261127):
    """HotpotQA train bridge questions in the bridge format, for training only.

    Skips questions whose supporting documents appear in any validation bridge case or the pinned test
    split, so no evaluated passage is trained on.
    """
    held = {document_key(t) for case in bridge_cases() for t in support_titles(case)}
    held |= {doc for row in split_rows('test') for doc in row.get('documents', [])}
    cases = [bridge_case(row) for row in hotpot_rows(HOTPOT_TRAIN)
             if not {document_key(t) for t in row['supporting_facts']['title']} & held]
    cases = [case for case in cases if case]
    random.Random(seed).shuffle(cases)
    return cases


def support_titles(case):
    """The two passage titles of a bridge case."""
    first = case.prompt.split('Passage 1:\n', 1)[1].split('\n', 1)[0]
    second = case.prompt.split('\n\nPassage 2:\n', 1)[1].split('\n', 1)[0]
    return first, second
