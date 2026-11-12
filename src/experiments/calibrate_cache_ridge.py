# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Calibrate a three-layer ridge cache map at Qwen's question-token queries."""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

import paired_kv_dataset as base
from cache_attention import CacheAttention
from cache_ridge import ThreeLayerRidge
from train_cache_attention import Dataset, causal_positions, checkpoint, translate_prefix


class RidgeCorrection(nn.Module):
    def __init__(self, width, layers, rank, device):
        super().__init__()
        self.down = nn.Parameter(torch.randn(2, layers, width, rank, device=device) * 0.02)
        self.up = nn.Parameter(torch.zeros(2, layers, rank, width, device=device))

    def forward(self, normalized):
        return normalized + torch.einsum(
            'trli,rlio->trlo', torch.einsum('trlw,rlwi->trli', normalized, self.down), self.up)

    @torch.no_grad()
    def fold(self, weights):
        return weights + (weights @ self.down) @ self.up


@torch.no_grad()
def capture_queries(model, tokens, probes, heads, dim):
    queries, handles = [], []

    def record(module, inputs, output):
        queries.append(output[0, probes].reshape(len(probes), heads, dim).transpose(0, 1))

    try:
        for layer in model.model.layers:
            handles.append(layer.self_attn.q_proj.register_forward_hook(record))
        model(input_ids=tokens, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(queries)


def attention_loss(query, probes, native, predicted, cos, sin, output_weights):
    layers, heads, _, dim = query.shape
    tokens = len(native)
    kv_heads = native.shape[2] // layers
    groups = heads // kv_heads

    def separate(features):
        keys, values = features.reshape(tokens, 2, layers, kv_heads, dim).permute(1, 2, 3, 0, 4)
        keys = apply_rotary_pos_emb(keys, keys, cos, sin)[0]
        return keys.repeat_interleave(groups, dim=1), values.repeat_interleave(groups, dim=1)

    query = apply_rotary_pos_emb(query, query, cos[:, probes], sin[:, probes])[0]
    native_key, native_value = separate(native)
    mapped_key, mapped_value = separate(predicted)
    future = torch.arange(tokens, device=query.device) > probes[:, None]

    def pattern(key):
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(dim)
        return scores.masked_fill(future, torch.finfo(scores.dtype).min).log_softmax(-1)

    native_logp, mapped_logp = pattern(native_key), pattern(mapped_key)
    native_p, mapped_p = native_logp.exp(), mapped_logp.exp()
    native_read = native_p @ native_value
    key_difference = (mapped_p @ native_value) - native_read
    key_output_error = (key_difference @ output_weights).square().mean()
    key_kl = (native_p * (native_logp - mapped_logp)).sum(dim=-1).mean()
    native_output = torch.einsum('lhpd,lhdw->lpw', native_read, output_weights)
    mapped_output = torch.einsum('lhpd,lhdw->lpw', mapped_p.detach() @ mapped_value, output_weights)
    value_error = ((mapped_output - native_output).square().sum(dim=(1, 2)) /
                   native_output.square().sum(dim=(1, 2)).clamp_min(1e-8)).mean()
    return torch.stack((key_kl, key_output_error, value_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--checkpoint-file', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--pilot', action='store_true', help='Calibrate on eight prompts and validate on four')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error('Expose one allocated CUDA GPU')
    args.work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    randomizer = random.Random(20261111)
    saved = torch.load(args.checkpoint_file, map_location='cuda', weights_only=True)
    run = saved['run']
    if (run['source'], run['target']) != ('smol', 'qwen'):
        raise ValueError('Expected a SmolLM-to-Qwen checkpoint')
    dataset_args = SimpleNamespace(local_data=None, data_root=args.data_root, source='smol',
                                   target='qwen', pilot=False, device='cuda', batch=128)
    data = Dataset(dataset_args)
    if (data.manifest_sha != run['manifest_sha256'] or [s['id'] for s in data.train] != run['train_ids']
            or [s['id'] for s in data.validation] != run['validation_ids']):
        raise ValueError('Dataset differs from checkpoint')
    baseline = CacheAttention(run['config']).to('cuda').eval()
    baseline.load_state_dict(saved['model'])
    ridge = ThreeLayerRidge(baseline)
    ridge.fit(data, run['stride'], run['ridge'])
    correction = RidgeCorrection(ridge.target_width, len(ridge.layers), 16, 'cuda')
    optimizer = torch.optim.AdamW(correction.parameters(), lr=1e-3)
    spec = base.MODELS['qwen']
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
                                               dtype=torch.float32, attn_implementation='eager').eval().cuda()
    model.requires_grad_(False)
    heads, dim = model.config.num_attention_heads, run['config']['head_dim']
    weights = torch.stack([layer.self_attn.o_proj.weight.T.reshape(heads, dim, -1)
                           for layer in model.model.layers])
    train = data.train[:8] if args.pilot else list(data.train)
    validation = data.validation[:4] if args.pilot else data.validation
    traces = {}
    history = []
    started = time.perf_counter()

    def loss(sample):
        probes = torch.tensor(sample['question_positions']['qwen'], device='cuda')
        with torch.no_grad():
            tokens = torch.tensor([sample['tokens']['qwen']['ids']], device='cuda')
            if sample['id'] not in traces:
                traces[sample['id']] = capture_queries(model, tokens, probes, heads, dim).cpu()
            query = traces[sample['id']].cuda()
            length = int(probes[-1]) + 1
            source = data.features(sample, 'smol')
            native = data.features(sample, 'qwen')[:length].cuda()
            mapped = translate_prefix(ridge, source, sample, dataset_args, length, True)
            cos, sin = model.model.rotary_emb(native, torch.arange(length, device='cuda')[None])
            normalized = baseline.standardize_target(mapped).reshape(length, 2, len(ridge.layers), -1)
            _, valid = causal_positions(sample, 'smol', 'qwen')
            valid = valid[:length].cuda().reshape(-1, 1, 1, 1)
        # Preserve the mean fallback for positions without a causal source match.
        corrected = torch.where(valid, correction(normalized), normalized).reshape_as(mapped)
        predicted = corrected * baseline.target_std + baseline.target_mean
        return attention_loss(query, probes, native, predicted, cos, sin, weights)

    @torch.no_grad()
    def validate():
        losses = torch.stack([loss(s) for s in validation]).mean(0).tolist()
        if not all(math.isfinite(v) for v in losses):
            raise ValueError('Nonfinite validation loss')
        return dict(zip(('attention_kl', 'key_output_mse', 'value_relative_mse'), losses))

    def record(epoch, train_loss):
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'validation': validate()}
        history.append(metrics)
        base.write_json(args.work / 'training.json', history)
        print(json.dumps(metrics), flush=True)
        return sum(metrics['validation'].values())

    best = record(0, None)
    best_epoch = 0
    best_weights = ridge.weights
    for epoch in range(1, (2 if args.pilot else 4) + 1):
        randomizer.shuffle(train)
        losses = []
        for sample in train:
            optimizer.zero_grad(set_to_none=True)
            total = loss(sample).sum()
            if not torch.isfinite(total):
                raise ValueError(f'Nonfinite calibration loss for {sample["id"]}')
            total.backward()
            nn.utils.clip_grad_norm_(correction.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(total.item())
        score = record(epoch, statistics.mean(losses))
        if score < best:
            best, best_epoch, best_weights = score, epoch, correction.fold(ridge.weights)

    # Check the final factors and folded map agree, including the intercept row.
    x = ridge.design(data.features(validation[0], 'smol')[:16].cuda())
    with torch.no_grad():
        plain = torch.einsum('trli,rlio->trlo', x, ridge.weights)
        folded = torch.einsum('trli,rlio->trlo', x, correction.fold(ridge.weights))
        torch.testing.assert_close(folded, correction(plain), atol=2e-5, rtol=1e-4)
    checkpoint(args.work / 'folded-ridge.pt', {
        'parent_sha256': base.file_sha(args.checkpoint_file), 'manifest_sha256': data.manifest_sha,
        'ridge': ridge.weights.cpu(), 'calibrated': best_weights.cpu(), 'source_layers': ridge.layers.cpu(),
        'best_epoch': best_epoch, 'rank': 16, 'train_prompts': len(train),
        'validation_prompts': len(validation), 'pilot': args.pilot,
        'trainable_parameters': sum(p.numel() for p in correction.parameters()),
        'seconds': time.perf_counter() - started})
    print(f'COMPLETE: selected epoch {best_epoch}; folded map saved to {args.work}', flush=True)


if __name__ == '__main__':
    main()
