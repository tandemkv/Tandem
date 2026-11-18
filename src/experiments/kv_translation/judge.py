"""LLM judge for whether a translated answer says the same thing as native Qwen's answer."""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from metrics import ARTICLES, ENTITY, OPENERS, first_sentence

JUDGE = ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
PROMPT = """Question: {question}

Answer A: {native}

Answer B: {candidate}

Do A and B give the same answer to the question? Compare only the answer each one gives to the question, \
which must name the same entities, numbers, and dates. Ignore everything else either one says: one may be \
shorter, add explanation, or be cut off mid-sentence. Reply with exactly SAME or DIFFERENT."""


class Judge:
    def __init__(self, device='cuda'):
        # 4-bit weights (about 20 GB) fit on one shared GPU. Keep the judge on a single GPU:
        # splitting it across ada's L40S cards corrupted GPU-to-GPU copies.
        name, revision = JUDGE
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                   bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'': 0 if device == 'cuda' else device}).eval()
        self.same, self.different = (self.tokenizer.encode(w, add_special_tokens=False)[0]
                                     for w in ('SAME', 'DIFFERENT'))

    @torch.inference_mode()
    def __call__(self, question, native, candidate):
        """True when the judge prefers SAME over DIFFERENT for the first reply token."""
        content = PROMPT.format(question=question, native=native, candidate=candidate)
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                 add_generation_prompt=True, return_tensors='pt')
        logits = self.model(input_ids=ids.to(self.model.device), logits_to_keep=1).logits[0, -1]
        if not torch.isfinite(logits).all():  # Broken GPU-to-GPU copies on ada produced NaN logits.
            raise RuntimeError('Judge produced non-finite logits')
        return bool(logits[self.same] > logits[self.different])


def annotate(rows, judge):
    """Add the judge verdict and semantic agreement (judge SAME and matching entities) to each row."""
    for row in rows:
        for result in row['conditions'].values():
            metrics = result['metrics']
            metrics['judge'] = judge(row['question'], row['native']['answer'], result['answer'])
            metrics['agree'] = metrics['judge'] and metrics['entities']
    return rows


def answer_entity(answer, question):
    """The first entity in the answer's first sentence that the question does not already name."""
    for match in ENTITY.finditer(first_sentence(answer)):
        words = match.group()
        if not set(words.lower().split()) <= ARTICLES | OPENERS | {'yes', 'no'} \
                and words.lower() not in question.lower():
            return words
    return None


def validation_pairs(rows):
    """Pairs with known verdicts built from native answers: (question, a, b, expected_same).

    SAME: the full answer versus its own first sentence. DIFFERENT: a leading yes/no flipped, or
    else every mention of the answer entity replaced by another case's answer entity.
    """
    pairs = []
    for row, other in zip(rows, rows[1:] + rows[:1]):
        question, answer = row['question'], row['native']['answer']
        pairs.append((question, answer, first_sentence(answer), True))
        lead = re.match(r'\s*(Yes|No)\b', answer)
        if lead:
            flipped = 'No' if lead.group(1) == 'Yes' else 'Yes'
            pairs.append((question, answer, answer[:lead.start(1)] + flipped + answer[lead.end(1):], False))
            continue
        mine = answer_entity(answer, question)
        theirs = answer_entity(other['native']['answer'], other['question'])
        if mine and theirs and mine.lower() != theirs.lower():
            pairs.append((question, answer, answer.replace(mine, theirs), False))
    return pairs


def validate(rows, judge):
    """Judge accuracy on pairs with known verdicts, split by expected verdict."""
    hits, misses = {True: [], False: []}, []
    for question, a, b, same in validation_pairs(rows):
        hits[same].append(judge(question, a, b) == same)
        if not hits[same][-1]:
            misses.append({'question': question, 'a': a, 'b': b, 'expected_same': same})
    return {'same_accuracy': sum(hits[True]) / len(hits[True]), 'same_pairs': len(hits[True]),
            'different_accuracy': sum(hits[False]) / len(hits[False]), 'different_pairs': len(hits[False]),
            'misses': misses}
