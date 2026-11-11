# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Score saved cache-handoff answers for equivalence and gold correctness."""
import argparse
import json
from itertools import batched
from pathlib import Path
from statistics import mean

BEM = 'kortukov/answer-equivalence-bem'
REVISION = 'b7216e6d26bbe6e7a8da2bec5b00043930ec55ae'
THRESHOLD = 0.5


def question_from_prompt(prompt):
    return prompt.rsplit('\n\nQuestion: ', 1)[1].removesuffix('\nAnswer:')


def bem_scores(pairs):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BEM, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BEM, revision=REVISION,
                                                                use_safetensors=True).eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(pairs), 16):
            batch = pairs[start:start + 16]
            inputs = tokenizer(
                [f'[CLS] {candidate} [SEP]' for _, _, candidate in batch],
                [f'{reference} [SEP] {question} [SEP]' for question, reference, _ in batch],
                add_special_tokens=False, padding=True, truncation=True, max_length=512, return_tensors='pt',
            )
            scores.extend(model(**inputs).logits.softmax(dim=-1)[:, 1].tolist())
    return [score if reference.strip() and candidate.strip() else 0.0
            for score, (_, reference, candidate) in zip(scores, pairs, strict=True)]


def evaluate(rows, method, score=bem_scores):
    if method == 'native' or not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected a handoff method and a nonempty evaluation with unique prompt IDs')
    pairs, details = [], []
    for row in rows:
        question, gold = row['question'], row['reference']
        native, handoff = row['native']['answer'], row[method]['answer']
        if not question.strip() or not gold.strip():
            raise ValueError(f'Missing question or gold answer for {row["id"]}')
        pairs.extend((question, reference, candidate) for reference, candidate in
                     ((native, handoff), (handoff, native), (gold, native), (gold, handoff)))
        details.append(dict(id=row['id'], question=question, gold=gold, native=native, handoff=handoff))
    judgments = batched(score(pairs), 4)
    for row, (forward, reverse, native_gold, handoff_gold) in zip(details, judgments, strict=True):
        row.update(native_to_handoff=forward, handoff_to_native=reverse,
                   native_gold_score=native_gold, handoff_gold_score=handoff_gold,
                   agreement=min(forward, reverse) >= THRESHOLD,
                   native_correct=native_gold >= THRESHOLD, handoff_correct=handoff_gold >= THRESHOLD)
    native_correct = sum(row['native_correct'] for row in details)
    preserved = sum(row['native_correct'] and row['handoff_correct'] for row in details)
    fixed = sum(not row['native_correct'] and row['handoff_correct'] for row in details)
    return {
        'summary': {
            'prompts': len(details), 'agreement': mean(row['agreement'] for row in details),
            'native_accuracy': native_correct / len(details),
            'handoff_accuracy': (preserved + fixed) / len(details),
            'native_correct': native_correct, 'preserved': preserved,
            'lost': native_correct - preserved, 'fixed': fixed,
            'retention': preserved / native_correct if native_correct else None,
        },
        'rows': details,
    }


def main():
    from paired_kv_dataset import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--method', required=True)
    parser.add_argument('--test-data', type=Path, help='Test Parquet for older reports without question text')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.evaluation.read_text())
    if args.test_data:
        import pyarrow.parquet as pq

        test = {row['id']: row for row in pq.read_table(
            args.test_data, columns=['id', 'prompt', 'gold_answer']).to_pylist()}
        for row in rows:
            sample = test[row['id']]
            if row['reference'] != sample['gold_answer']:
                raise ValueError(f'Test reference differs for {row["id"]}')
            row.setdefault('question', question_from_prompt(sample['prompt']))
    if not all('question' in row and 'answer' in row.get('native', {})
               and 'answer' in row.get(args.method, {}) for row in rows):
        parser.error('Expected question and answer text. Use --test-data for older reports with saved answers.')
    if args.output.exists():
        parser.error('Choose a new output path')
    report = evaluate(rows, args.method)
    report.update(method=args.method, judge=BEM, revision=REVISION, threshold=THRESHOLD)
    write_json(args.output, report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
