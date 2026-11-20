"""Score answer equivalence against native Qwen and gold independently."""
from itertools import batched

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BEM = 'kortukov/answer-equivalence-bem'
REVISION = 'b7216e6d26bbe6e7a8da2bec5b00043930ec55ae'


@torch.inference_mode()
def bem_scores(pairs):
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BEM, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BEM, revision=REVISION,
                                                                use_safetensors=True).eval()
    for batch in batched(pairs, 16):
        inputs = tokenizer(
            [f'[CLS] {candidate} [SEP]' for _, _, candidate in batch],
            [f'{reference} [SEP] {question} [SEP]' for question, reference, _ in batch],
            add_special_tokens=False, padding=True, truncation=True, max_length=512, return_tensors='pt',
        )
        probabilities = model(**inputs).logits.softmax(dim=-1)[:, 1].tolist()
        for p, (_, reference, candidate) in zip(probabilities, batch, strict=True):
            yield p if reference.strip() and candidate.strip() else 0.0


def score_answers(rows, method, score=bem_scores):
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected a nonempty set of unique prompts')
    pairs = []
    for row in rows:
        question, gold = row['question'], row['reference']
        native, candidate = row['native']['answer'], row[method]['answer']
        if not question.strip() or not gold.strip():
            raise ValueError(f'Missing question or gold answer for {row["id"]}')
        pairs.extend((question, reference, answer) for reference, answer in
                     ((native, candidate), (candidate, native), (gold, native), (gold, candidate)))
    agree = native_correct = candidate_correct = preserved = 0
    for _, (forward, reverse, native_gold, candidate_gold) in zip(rows, batched(score(pairs), 4), strict=True):
        agree += min(forward, reverse) >= 0.5
        native_correct += native_gold >= 0.5
        candidate_correct += candidate_gold >= 0.5
        preserved += min(native_gold, candidate_gold) >= 0.5
    return {'prompts': len(rows), 'agreement': agree / len(rows),
            'native_accuracy': native_correct / len(rows), 'handoff_accuracy': candidate_correct / len(rows),
            'native_correct': native_correct, 'preserved': preserved, 'lost': native_correct - preserved,
            'fixed': candidate_correct - preserved}
