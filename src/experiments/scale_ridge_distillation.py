# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Compare calibration data sizes at the same continuation-distillation budget."""
import argparse
from collections import Counter
import json
import random
import re
import statistics
import string
import time
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import paired_kv_dataset as base
from cache_attention import CacheAttention
from cache_ridge import ThreeLayerRidge
from calibrate_cache_attention import continuation_inputs, teacher_logits
from calibrate_cache_ridge import RidgeCorrection
from evaluate_cache_attention import cached_start, continue_greedy, native_start, prefix_agreement, timed
from train_cache_attention import Dataset, causal_positions, checkpoint, logit_kl, receiver_logits, stack_features, translate_prefix

SEED = 20261111
TRAIN_TOKENS = 8
EVAL_TOKENS = 32
LR = 1e-3


def rows(split, rng):
    files = [base.SOURCE['file'], 'distractor/train-00001-of-00002.parquet'] if split == 'train' else [
        'distractor/validation-00000-of-00001.parquet']
    for filename in files:
        path = hf_hub_download(base.SOURCE['id'], filename, revision=base.SOURCE['revision'],
                               repo_type='dataset')
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=512, columns=['id', 'question', 'context', 'answer']):
            samples = batch.to_pylist()
            rng.shuffle(samples)
            yield from samples


def select(split, count, tokenizers, max_tokens, used_ids, used_documents, used_text):
    selected = []
    for row in rows(split, random.Random(SEED)):
        docs = {base.document_key(title) for title in row['context']['title']}
        if row['id'] in used_ids or docs & used_documents:
            continue
        text, _ = base.format_prompt(row)
        sha = base.sha(text.encode())
        if sha in used_text:
            continue
        tokens = {alias: base.encode(tok, text) for alias, tok in tokenizers.items()}
        if any(not 64 <= len(t['ids']) <= max_tokens for t in tokens.values()):
            continue
        correspondence = base.align(tokens['smol']['offsets'], tokens['qwen']['offsets'])
        sample = {'id': row['id'], 'text_sha256': sha, 'documents': sorted(docs), 'tokens': tokens,
                  'alignment': {'smol_to_qwen': correspondence}}
        if split == 'validation':
            sample['answer'] = row['answer']
            sample['question'] = row['question'].strip()
        selected.append(sample)
        used_ids.add(row['id'])
        used_documents.update(docs)
        used_text.add(sha)
        if len(selected) == count:
            return selected
    raise ValueError(f'Found {len(selected)} of {count} disjoint {split} prompts')


def load_model(alias):
    spec = base.MODELS[alias]
    model = AutoModelForCausalLM.from_pretrained(
        spec['id'], revision=spec['revision'], dtype=torch.float32, attn_implementation='eager')
    return model.eval().cuda().requires_grad_(False)


def ids_for(sample):
    return torch.tensor([sample['tokens']['qwen']['ids']], device='cuda')


def source_features(smol, sample):
    tensors, geometry = base.capture(smol, sample['tokens']['smol']['ids'], 'cuda')
    return stack_features(tensors, geometry)


@torch.no_grad()
def prepare_training(data, existing, extra, smol, qwen, ridge, eos, traces, device_args):
    for index, sample in enumerate(existing + extra):
        out = traces / (sample['id'] + '.pt')
        source = data.features(sample, 'smol') if index < len(existing) else source_features(smol, sample)
        ids = ids_for(sample)
        length = ids.shape[1] - 1
        predicted = translate_prefix(ridge, source, sample, device_args, length, True)
        normalized = ridge.baseline.standardize_target(predicted).reshape(length, 2, len(ridge.layers), -1)
        native = native_start(qwen, ids)
        continuation = continue_greedy(qwen, native, TRAIN_TOKENS, eos)
        teacher = teacher_logits(qwen, ids, continuation)
        _, suffix = continuation_inputs(ids, continuation)
        _, valid = causal_positions(sample, 'smol', 'qwen')
        checkpoint(out, {'normalized': normalized.cpu(), 'teacher': teacher.cpu(),
                         'suffix': suffix.cpu(), 'valid': valid[:-1]})
        if (index + 1) % 40 == 0 or index + 1 == len(existing) + len(extra):
            print(f'Prepared {index + 1}/{len(existing) + len(extra)} training traces', flush=True)


def train(qwen, ridge, samples, traces, steps, out):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    correction = RidgeCorrection(ridge.target_width, len(ridge.layers), 16, 'cuda')
    optimizer = torch.optim.AdamW(correction.parameters(), lr=LR)
    completed = 0
    running = []
    while completed < steps:
        order = list(samples)
        rng.shuffle(order)
        for sample in order[:steps - completed]:
            item = torch.load(traces / (sample['id'] + '.pt'), map_location='cuda', weights_only=True)
            normalized = item['normalized']
            valid = item['valid'].reshape(-1, 1, 1, 1)
            corrected = torch.where(valid, correction(normalized), normalized)
            shape = (len(normalized), *ridge.target_mean.shape)
            features = corrected.reshape(shape) * ridge.baseline.target_std + ridge.target_mean
            optimizer.zero_grad(set_to_none=True)
            student = receiver_logits(qwen, features, item['suffix'])
            loss = logit_kl(student, item['teacher'])
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss on step {completed}')
            loss.backward()
            nn.utils.clip_grad_norm_(correction.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed += 1
            running.append(loss.item())
            if completed % 80 == 0 or completed == steps:
                print(f'{len(samples)} prompts: {completed}/{steps} steps, mean KL {statistics.mean(running):.4f}',
                      flush=True)
                running.clear()
    weights = correction.fold(ridge.weights)
    with torch.no_grad():
        example = torch.randn(3, *ridge.weights.shape[:-1], device='cuda')
        example[..., -1] = 1
        original = torch.einsum('trli,rlio->trlo', example, ridge.weights)
        folded = torch.einsum('trli,rlio->trlo', example, weights)
        torch.testing.assert_close(correction(original), folded, atol=2e-5, rtol=1e-4)
    checkpoint(out, {'weights': weights.cpu(), 'train_ids': [s['id'] for s in samples],
                     'steps': steps, 'lr': LR, 'rank': 16, 'native_continuation_tokens': TRAIN_TOKENS})
    return weights


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_score(text, reference):
    answer = normalize(text)
    gold = normalize(reference)
    exact = answer == gold
    if not exact and {answer, gold} & {'yes', 'no', 'noanswer'}:
        return {'exact_match': exact, 'f1': 0.0}
    a, b = answer.split(), gold.split()
    common = sum((Counter(a) & Counter(b)).values())
    return {'exact_match': exact, 'f1': 2 * common / (len(a) + len(b)) if a or b else 0.0}


def score_generation(tokenizer, tokens, native_tokens, reference):
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    first_line = answer.split('\n', 1)[0]
    return dict(answer_score(first_line, reference), answer=answer, **prefix_agreement(native_tokens, tokens))


@torch.inference_mode()
def evaluate(qwen, smol, tokenizer, ridge, variants, test, eos, device_args, out):
    results = []
    for index, sample in enumerate(test):
        ids = ids_for(sample)
        source = source_features(smol, sample)
        native = native_start(qwen, ids)
        native_tokens = continue_greedy(qwen, native, EVAL_TOKENS, eos)
        # Control verifies the same native cache injection used by the translators.
        tensors, geometry = base.capture(qwen, sample['tokens']['qwen']['ids'][:-1], 'cuda')
        control = cached_start(qwen, stack_features(tensors, geometry).cuda(), ids[:, -1:])
        torch.testing.assert_close(control.logits, native.logits, rtol=1e-3, atol=1e-3)

        def score(tokens):
            return score_generation(tokenizer, tokens, native_tokens, sample['answer'])

        row = {'id': sample['id'], 'question': sample['question'], 'reference': sample['answer'],
               'prompt_tokens': ids.shape[1],
               'native': score(native_tokens)}
        row['native']['ms'] = statistics.median(timed(lambda: native_start(qwen, ids), 'cuda', 3))
        for name, weights in variants.items():
            ridge.weights = weights

            def start():
                predicted = translate_prefix(ridge, source, sample, device_args, ids.shape[1] - 1, True)
                return cached_start(qwen, predicted, ids[:, -1:])

            predicted = start()
            kl = logit_kl(predicted.logits, native.logits).item()
            row[name] = dict(score(continue_greedy(qwen, predicted, EVAL_TOKENS, eos)), kl=kl,
                             ms=statistics.median(timed(start, 'cuda', 3)))
        results.append(row)
        base.write_json(out, results)
        if (index + 1) % 20 == 0 or index + 1 == len(test):
            print(f'Evaluated {index + 1}/{len(test)} untouched prompts', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--attention-checkpoint', type=Path, required=True)
    parser.add_argument('--ridge-checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error('Expose one allocated CUDA GPU')
    args.work.mkdir(parents=True, exist_ok=False)
    traces = args.work / 'traces'
    traces.mkdir()
    torch.set_num_threads(4)
    started = time.perf_counter()
    small, large, test_count, steps = (4, 8, 4, 8) if args.pilot else (160, 800, 120, 800)
    data_args = SimpleNamespace(local_data=None, data_root=args.data_root, source='smol',
                                target='qwen', pilot=False, device='cuda', batch=128)
    data = Dataset(data_args)
    parent = torch.load(args.attention_checkpoint, map_location='cuda', weights_only=True)
    saved_ridge = torch.load(args.ridge_checkpoint, map_location='cuda', weights_only=True)
    if (saved_ridge['parent_sha256'] != base.file_sha(args.attention_checkpoint)
            or saved_ridge['manifest_sha256'] != data.manifest_sha
            or [s['id'] for s in data.train] != parent['run']['train_ids']):
        raise ValueError('Ridge map and training manifest do not match')
    baseline = CacheAttention(parent['run']['config']).to('cuda').eval()
    baseline.load_state_dict(parent['model'])
    ridge = ThreeLayerRidge(baseline)
    if not torch.equal(ridge.layers, saved_ridge['source_layers']):
        raise ValueError('Ridge layer selection differs')
    ridge.weights = saved_ridge['ridge']
    variants = {'plain_ridge': ridge.weights}
    tokenizers = {key: AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
                  for key, spec in base.MODELS.items()}
    original = data.train + data.validation
    used_ids = {s['id'] for s in original}
    used_docs = {d for s in original for d in s['documents']}
    used_text = {s['text_sha256'] for s in original}
    # Reserve test documents before selecting added training prompts.
    reserved_test = select('validation', 124, tokenizers, data.manifest['max_tokens'], used_ids, used_docs, used_text)
    test = reserved_test[-test_count:] if args.pilot else reserved_test[:test_count]
    extra = select('train', large - small, tokenizers, data.manifest['max_tokens'], used_ids, used_docs, used_text)
    existing = data.train[:small]
    base.write_json(args.work / 'split.json', {
        'source': base.SOURCE, 'seed': SEED, 'original_manifest_sha256': data.manifest_sha,
        'original_train_ids': [s['id'] for s in existing], 'added': extra, 'test': test,
        'ridge_sha256': base.file_sha(args.ridge_checkpoint), 'steps_per_arm': steps,
        'training_tokens': TRAIN_TOKENS, 'evaluation_tokens': EVAL_TOKENS, 'pilot': args.pilot})
    print(f'Split ready: {small} vs {large} training prompts, {test_count} untouched prompts', flush=True)
    smol, qwen = load_model('smol'), load_model('qwen')
    eos = qwen.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prepare_training(data, existing, extra, smol, qwen, ridge, eos, traces, data_args)
    for label, samples in ((f'distilled_{small}', existing), (f'distilled_{large}', existing + extra)):
        variants[label] = train(qwen, ridge, samples, traces, steps, args.work / (label + '.pt'))
    data_args.batch = parent['run']['batch']
    results = evaluate(qwen, smol, tokenizers['qwen'], ridge, variants, test, eos, data_args, args.work / 'evaluation.json')
    summary = {}
    for name in ('native', *variants):
        values = [row[name] for row in results]
        summary[name] = {key: statistics.mean(v[key] for v in values)
                         for key in ('exact_match', 'f1', 'first_token_match', 'matching_prefix_tokens', 'sequence_equal')}
        summary[name]['median_ms'] = statistics.median(v['ms'] for v in values)
        if name != 'native':
            summary[name]['first_token_kl'] = statistics.mean(v['kl'] for v in values)
    report = {'summary': summary, 'test_prompts': len(results), 'seconds': time.perf_counter() - started,
              'timing_scope': 'CPU source features to target first logits, excluding source prefill and tokenization'}
    base.write_json(args.work / 'summary.json', report)
    print('COMPLETE: ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
