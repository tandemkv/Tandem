"""LLM judge for whether a translated answer says the same thing as native Qwen's answer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from metrics import ARTICLES, ENTITY, OPENERS, first_sentence

JUDGE = ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
PROMPT = """Question: {question}

Answer A: {native}

Answer B: {candidate}

Do A and B give the same answer to the question? They must name the same entities, numbers, and dates. \
Wording, length, and extra explanation do not matter. Either answer may be cut off mid-sentence. \
Reply with exactly SAME or DIFFERENT."""


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


def first_entity(text):
    for match in ENTITY.finditer(first_sentence(text)):
        if not set(match.group().lower().split()) <= ARTICLES | OPENERS:
            return match.group()
    return None


def validation_pairs(rows):
    """Pairs with known verdicts built from native answers: (question, a, b, expected_same).

    SAME: the full answer versus its own first sentence. DIFFERENT: the answer with its first
    entity replaced by another case's first entity.
    """
    pairs = []
    for row, other in zip(rows, rows[1:] + rows[:1]):
        answer = row['native']['answer']
        pairs.append((row['question'], answer, first_sentence(answer), True))
        mine, theirs = first_entity(answer), first_entity(other['native']['answer'])
        if mine and theirs and mine.lower() != theirs.lower():
            pairs.append((row['question'], answer, answer.replace(mine, theirs, 1), False))
    return pairs


def validate(rows, judge):
    """Judge accuracy on pairs with known verdicts, split by expected verdict."""
    hits = {True: [], False: []}
    for question, a, b, same in validation_pairs(rows):
        hits[same].append(judge(question, a, b) == same)
    return {'same_accuracy': sum(hits[True]) / len(hits[True]), 'same_pairs': len(hits[True]),
            'different_accuracy': sum(hits[False]) / len(hits[False]), 'different_pairs': len(hits[False])}
