"""LLM judge for whether a translated answer says the same thing as native Qwen's answer."""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from metrics import ARTICLES, ENTITY, OPENERS, first_sentence

JUDGE = ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
PROMPT = """You check whether two responses give the same answer to a question.

Rules:
- Find the core answer each response gives to the question: the entity, number, date, place, or yes/no.
- SAME if the core answers match, even when one response is longer, adds explanation or extra facts, \
names the answer more or less fully, or is cut off after giving it.
- DIFFERENT if the core answers differ, or if one response never gives an answer.

Examples:
- "Paris." vs "The capital is Paris, which is also its largest city." -> SAME
- "Universal Pictures" vs "Universal." -> SAME
- "Yes, both are lakes in Canada." vs "Yes." -> SAME
- "Born in 1952." vs "He was born in 1957 in Ohio." -> DIFFERENT
- "Lyon." vs "The answer is" -> DIFFERENT

Question: {question}

Response A: {native}

Response B: {candidate}

In one short sentence, state the core answer each response gives. \
Then end with exactly "Verdict: SAME" or "Verdict: DIFFERENT"."""
VERDICT = re.compile(r'Verdict:\s*\**\s*(SAME|DIFFERENT)', re.IGNORECASE)
BATCH, MAX_NEW_TOKENS = 8, 96  # Small batches leave room on shared GPUs.


def parse_verdict(reply):
    """True for a final SAME verdict. A missing verdict counts as not SAME."""
    verdicts = VERDICT.findall(reply)
    return bool(verdicts) and verdicts[-1].upper() == 'SAME'


class Judge:
    def __init__(self, device='cuda'):
        # 4-bit weights (about 20 GB) fit on one shared GPU. Keep the judge on a single GPU:
        # splitting it across ada's L40S cards corrupted GPU-to-GPU copies.
        name, revision = JUDGE
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                   bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'': 0 if device == 'cuda' else device}).eval()

    @torch.inference_mode()
    def __call__(self, triples):
        """Judge (question, native, candidate) triples in batches. Returns (same, reply) per triple.

        A reply without a parsable verdict counts as not SAME.
        """
        results = []
        for i in range(0, len(triples), BATCH):
            texts = [self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': PROMPT.format(question=q, native=a, candidate=b)}],
                tokenize=False, add_generation_prompt=True) for q, a, b in triples[i:i + BATCH]]
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         top_p=None, top_k=None, temperature=None)
            for reply in self.tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True):
                results.append((parse_verdict(reply), reply.strip()))
            print(f'judged {len(results)}/{len(triples)}', flush=True)
        return results


def annotate(rows, judge):
    """Add the judge verdict and semantic agreement (judge SAME and matching entities) to each row."""
    targets = [(row, result) for row in rows for result in row['conditions'].values()]
    verdicts = judge([(row['question'], row['native']['answer'], result['answer']) for row, result in targets])
    for (row, result), (same, reply) in zip(targets, verdicts, strict=True):
        result['judge_reply'] = reply
        result['metrics']['judge'] = same
        result['metrics']['agree'] = same and result['metrics']['entities']
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
    pairs = validation_pairs(rows)
    hits, misses = {True: [], False: []}, []
    for (question, a, b, same), (verdict, reply) in zip(pairs, judge([p[:3] for p in pairs]), strict=True):
        hits[same].append(verdict == same)
        if verdict != same:
            misses.append({'question': question, 'a': a, 'b': b, 'expected_same': same, 'reply': reply})
    return {'same_accuracy': sum(hits[True]) / len(hits[True]), 'same_pairs': len(hits[True]),
            'different_accuracy': sum(hits[False]) / len(hits[False]), 'different_pairs': len(hits[False]),
            'misses': misses}
