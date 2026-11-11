# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Compare fixed and validation-selected source layers for SmolLM2-to-Qwen ridge."""
import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

import paired_kv_dataset as base
from cache_attention import CacheAttention
from cache_ridge import ThreeLayerRidge
from evaluate_cache_attention import cached_start, continue_greedy, native_start, timed
from evaluate_qa_answers import question_from_prompt
from scale_ridge_distillation import EVAL_TOKENS, ids_for, load_model, score_generation, source_features
from train_cache_attention import Dataset, checkpoint, logit_kl, receiver_logits, stack_features, translate_prefix

PROBE_TOKENS = 16


def layer_statistics(data, samples, baseline, stride):
    config = baseline.config
    sl, tl = config['source_layers'], config['target_layers']
    dim = config['source_heads'] * config['head_dim'] + 1
    width = config['target_heads'] * config['head_dim']
    kwargs = {'device': baseline.source_mean.device, 'dtype': torch.float64}
    gram = torch.zeros(2 * sl, dim, dim, **kwargs)
    cross = torch.zeros(2, sl * dim, tl * width, **kwargs)
    total = torch.zeros(2, tl * width, **kwargs)
    square = torch.zeros_like(total)
    count = 0
    for sample in samples:
        source, target = (tensor.to(gram.device) for tensor in data.pair(sample, stride))
        n = len(source)
        x = baseline.standardize_source(source).reshape(n, 2, sl, dim - 1).double()
        x = F.pad(x, (0, 1), value=1).permute(1, 2, 3, 0).contiguous()
        y = baseline.standardize_target(target).reshape(n, 2, tl * width).double().transpose(0, 1)
        flat = x.flatten(0, 1)
        gram.baddbmm_(flat, flat.transpose(-2, -1))
        cross.baddbmm_(x.flatten(1, 2), y)
        total += y.sum(dim=1)
        square += y.square().sum(dim=1)
        count += n
    if count < 2:
        raise ValueError('Layer selection needs at least two positions')
    return gram, cross.reshape(2 * sl, dim, tl * width), total, square, count


@torch.inference_mode()
def select_layers(data, baseline, stride, penalty):
    config = baseline.config
    sl, tl = config['source_layers'], config['target_layers']
    gram, cross, _, _, count = layer_statistics(data, data.train, baseline, stride)
    gram.div_(count).diagonal(dim1=-2, dim2=-1)[..., :-1].add_(penalty)
    weights = torch.linalg.solve(gram, cross.div_(count))
    del gram, cross
    gram, cross, total, square, count = layer_statistics(data, data.validation, baseline, stride)
    errors = weights * (gram @ weights - 2 * cross)
    sse = errors.reshape(2, sl, gram.shape[-1], tl, -1).sum(dim=(2, 4))
    sse += square.reshape(2, 1, tl, -1).sum(dim=-1)
    sst = (square - total.square() / count).reshape(2, 1, tl, -1).sum(dim=-1)
    r2 = 1 - sse / sst
    if not torch.isfinite(r2).all():
        raise ValueError('Nonfinite validation layer scores')
    # Keep one ranked layer list per target layer, using equal weight for keys and values.
    return r2.mean(dim=0).T.argsort(dim=-1, descending=True, stable=True), r2


def load_test(path, data):
    manifest = json.loads(path.with_name('manifest.json').read_text())
    if (manifest['sha256']['test.parquet'] != base.file_sha(path)
            or manifest['original_manifest_sha256'] != data.manifest_sha
            or manifest['models'] != base.MODELS):
        raise ValueError('Test dataset does not match the training data or model pins')
    rows = pq.read_table(path).to_pylist()
    fitting = data.train + data.validation
    fit_docs = {doc for sample in fitting for doc in sample['documents']}
    fit_ids = {sample['id'] for sample in fitting}
    if (len(rows) != 120 or len({row['id'] for row in rows}) != 120
            or fit_ids.intersection(row['id'] for row in rows)
            or fit_docs.intersection(doc for row in rows for doc in row['document_hashes'])):
        raise ValueError('Expected the 120 previously held-out test prompts')
    for row in rows:
        text = row['prompt']
        passage, separator, _ = text.rpartition('\n\nQuestion: ')
        if not separator or base.sha(text.encode()) != row['prompt_sha256']:
            raise ValueError(f'Unexpected prompt text for {row["id"]}')
        passage_tokens = sum(0 < end <= len(passage) for _, end in row['target_offsets'])
        prefix = min(1022, 3 * passage_tokens // 4, passage_tokens - PROBE_TOKENS)
        if prefix < 32:
            raise ValueError(f'Passage is too short for the probe: {row["id"]}')
        yield {
            'id': row['id'], 'question': question_from_prompt(text),
            'answer': row['gold_answer'], 'probe_prefix': prefix,
            'tokens': {
                'smol': {'ids': row['source_token_ids'], 'offsets': row['source_offsets']},
                'qwen': {'ids': row['target_token_ids'], 'offsets': row['target_offsets']},
            },
            'alignment': {'smol_to_qwen': {'target_to_source': row['target_to_source']}},
        }


def losses(logits, labels):
    nll = F.cross_entropy(logits[0].float(), labels[0], reduction='none')
    return {'next_token_nll': nll[0].item(), 'continuation_nll': nll.mean().item()}


@torch.inference_mode()
def evaluate(qwen, smol, test, maps, batch, out):
    spec = base.MODELS['qwen']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
    eos = qwen.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    options = SimpleNamespace(source='smol', target='qwen', batch=batch, device='cuda')
    rows = []
    for sample in test:
        ids = ids_for(sample)
        source = source_features(smol, sample)
        native = native_start(qwen, ids)
        native_tokens = continue_greedy(qwen, native, EVAL_TOKENS, eos)
        tensors, geometry = base.capture(qwen, sample['tokens']['qwen']['ids'][:-1], 'cuda')
        original = stack_features(tensors, geometry).cuda()
        control = cached_start(qwen, original, ids[:, -1:])
        torch.testing.assert_close(control.logits, native.logits, rtol=1e-3, atol=1e-3)

        prefix = sample['probe_prefix']
        end = prefix + PROBE_TOKENS - 1
        suffix = ids[:, prefix - 1:end]
        labels = ids[:, prefix:end + 1]
        native_logits = qwen(input_ids=ids[:, :end], logits_to_keep=PROBE_TOKENS, use_cache=False).logits
        cold_logits = qwen(input_ids=suffix, use_cache=False).logits
        replay = receiver_logits(qwen, original[:prefix - 1], suffix)
        torch.testing.assert_close(replay, native_logits, rtol=1e-3, atol=1e-3)

        def qa_score(tokens):
            return score_generation(tokenizer, tokens, native_tokens, sample['answer'])

        row = {
            'id': sample['id'], 'question': sample['question'], 'reference': sample['answer'],
            'probe_prefix': prefix,
            'cold': losses(cold_logits, labels),
            'native': dict(losses(native_logits, labels), **qa_score(native_tokens),
                           ms=statistics.median(timed(lambda: native_start(qwen, ids), 'cuda', 3))),
        }
        for name, mapper in maps.items():
            translated = translate_prefix(mapper, source, sample, options, prefix - 1, True)
            probe = losses(receiver_logits(qwen, translated, suffix), labels)

            def start():
                mapped = translate_prefix(mapper, source, sample, options, ids.shape[1] - 1, True)
                return cached_start(qwen, mapped, ids[:, -1:])

            prediction = start()
            kl = logit_kl(prediction.logits, native.logits).item()
            row[name] = dict(probe, **qa_score(continue_greedy(qwen, prediction, EVAL_TOKENS, eos)),
                             first_token_kl=kl, ms=statistics.median(timed(start, 'cuda', 3)))
        rows.append(row)
        base.write_json(out, rows)
        if len(rows) % 20 == 0:
            print(f'Evaluated {len(rows)}/120 held-out prompts', flush=True)
    return rows


def summarize(rows, names):
    summary = {}
    for name in names:
        summary[name] = {
            key: (statistics.median if key == 'ms' else statistics.mean)(row[name][key] for row in rows)
            for key in rows[0][name] if key != 'answer'
        }
    for metric in ('next_token_nll', 'continuation_nll'):
        cold, native = (summary[name][metric] for name in ('cold', 'native'))
        if cold <= native:
            raise ValueError(f'Native context did not improve {metric} over the cold baseline')
        for score in summary.values():
            score[metric.replace('_nll', '_recovery')] = (cold - score[metric]) / (cold - native)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error('Expose one allocated CUDA GPU')
    torch.set_num_threads(4)
    args.work.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = Dataset(SimpleNamespace(local_data=None, data_root=args.data_root,
                                   source='smol', target='qwen', pilot=False))
    test = list(load_test(args.test, data))
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    run = parent['run']
    if (run['manifest_sha256'] != data.manifest_sha or run['source'] != 'smol' or run['target'] != 'qwen'
            or run['train_ids'] != [s['id'] for s in data.train]):
        raise ValueError('Checkpoint does not match the fixed model pair and training split')
    baseline = CacheAttention(run['config']).cuda().eval()
    baseline.load_state_dict(parent['model'])
    print(f'Selecting layers with {len(data.train)} fit and {len(data.validation)} validation prompts', flush=True)
    ranking, r2 = select_layers(data, baseline, run['stride'], run['ridge'])
    maps = {'fixed_3': ThreeLayerRidge(baseline),
            'selected_3': ThreeLayerRidge(baseline, ranking[:, :3]),
            'selected_12': ThreeLayerRidge(baseline, ranking[:, :12])}
    metadata = {}
    for name, mapper in maps.items():
        start = time.perf_counter()
        count = mapper.fit(data, run['stride'], run['ridge'])
        metadata[name] = {'training_positions': count, 'weights': mapper.weights.numel(),
                          'fit_seconds': time.perf_counter() - start, 'layers': mapper.layers.tolist()}
        checkpoint(args.work / f'{name}.pt', {'weights': mapper.weights.cpu(), 'layers': mapper.layers.cpu(),
                   'parent_sha256': base.file_sha(args.checkpoint), 'manifest_sha256': data.manifest_sha})
        print(f'Fit {name} in {metadata[name]["fit_seconds"]:.1f}s', flush=True)
    base.write_json(args.work / 'layer_selection.json', {'validation_r2': r2.tolist(), 'fits': metadata})
    torch.cuda.empty_cache()
    qwen, smol = load_model('qwen'), load_model('smol')
    rows = evaluate(qwen, smol, test, maps, run['batch'], args.work / 'evaluation.json')
    report = {'summary': summarize(rows, ('cold', 'native', *maps)), 'fits': metadata,
              'test_prompts': len(rows), 'probe_tokens': PROBE_TOKENS,
              'train_prompts': len(data.train), 'selection_prompts': len(data.validation),
              'test_sha256': base.file_sha(args.test), 'seconds': time.perf_counter() - started}
    base.write_json(args.work / 'summary.json', report)
    print('COMPLETE: ' + json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
