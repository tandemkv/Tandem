# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "accelerate==1.9.0", "bitsandbytes==0.50.2",
#   "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0",
# ]
# ///
"""Refit the key half of a map so native Qwen queries attend to translated keys as they do to native keys.

Each layer is scored on its own: queries come from a native run over prefix, question, and native answer,
so no gradient flows through Qwen. Values stay fixed. The loss is the attention KL from native keys to
translated prefix keys plus native suffix keys, summed over layers and averaged over heads and queries.

  train_keys.py --start distilled --steps 4000 --out DIR   -> DIR/keyfit.pt, DIR/log.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

import harness
from conditions import Prepared
from data import Case
from mapper import capture, translate

VALIDATION = 100  # The last hotpot training prompts, held out from the key fit.


def train_cases(seed=20261119):
    rows = [r for r in harness.split_rows('train') if r['dataset'] == 'hotpot' and '\n\nQuestion: ' in r['prompt']]
    cases = [Case(r['id'], r['prompt'], r['question'], r['answer']) for r in rows]
    fit, held = cases[:-VALIDATION], cases[-VALIDATION:]
    random.Random(seed).shuffle(fit)
    return fit, held


@torch.no_grad()
def native_queries(h, ids):
    """Rotated native queries and keys for every layer: [layer, head, token, dim]."""
    queries, handles = [], []
    for layer in h.qwen.model.layers:
        handles.append(layer.self_attn.q_proj.register_forward_hook(lambda _, __, out: queries.append(out[0])))
    try:
        keys = capture(h.qwen, ids)[:, 0]
    finally:
        for handle in handles:
            handle.remove()
    config = h.qwen.config
    dim = config.hidden_size // config.num_attention_heads
    q = torch.stack(queries).unflatten(-1, (config.num_attention_heads, dim)).transpose(1, 2)
    return rotate(h, q), keys


def rotate(h, x, start=0):
    """RoPE at positions start.. for [..., head, token, dim]."""
    positions = torch.arange(start, start + x.shape[-2], device=x.device)[None]
    cos, sin = h.qwen.model.rotary_emb(x, positions)
    rotated, _ = apply_rotary_pos_emb(x.flatten(0, -3)[None], x.flatten(0, -3)[None], cos, sin)
    return rotated[0].reshape(x.shape)


def unrotated_keys(h, keys):
    """[token, layer-head, dim] -> [layer, head, token, dim]."""
    config = h.qwen.config
    return keys.unflatten(1, (config.num_hidden_layers, config.num_key_value_heads)).permute(1, 2, 0, 3)


def attention(q, k):
    """Log attention of the last queries over all earlier keys, with grouped heads."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    rows = torch.arange(k.shape[2] - q.shape[2], k.shape[2], device=q.device)[:, None]
    scores = scores.masked_fill(torch.arange(k.shape[2], device=q.device) > rows, -1e9)
    return scores.log_softmax(-1)


class Example:
    """Source K/V and native queries and keys for one prompt. Held-out examples wait on the CPU."""

    @torch.no_grad()
    def __init__(self, h, case, device):
        p = Prepared(h, case)
        native, _ = h.generate(p.target_ids + p.suffix)
        ids = p.target_ids + p.suffix + native['tokens'][:-1]
        self.alignment, self.prefix = p.alignment, len(p.target_ids)
        queries, native_keys = native_queries(h, ids)
        keys = rotate(h, unrotated_keys(h, native_keys))
        self.tensors = [t.to(device) for t in (p.source, queries[:, :, self.prefix:], keys)]


def loss(h, example, weights, values):
    source, queries, keys = (t.to(h.device) for t in example.tensors)
    with torch.no_grad():
        target = attention(queries, keys)
    state = {'layers': h.checkpoints['distilled']['layers'], 'weights': torch.stack((weights, values))}
    translated = translate(source, example.alignment, state, h.normalization,
                           h.smol.config.num_key_value_heads)[:, 0]
    keys = torch.cat((rotate(h, unrotated_keys(h, translated)), keys[:, :, example.prefix:]), dim=2)
    kl = (target.exp() * (target - attention(queries, keys))).sum(-1)
    return kl.mean((1, 2)).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', choices=tuple(harness.CHECKPOINTS), default='distilled')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--every', type=int, default=500, help='Validate and save this often')
    parser.add_argument('--validation', type=int, default=VALIDATION)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h = harness.Harness(args.device)
    for model in h.models.values():
        model.requires_grad_(False)
    start = h.checkpoints[args.start]
    weights = torch.nn.Parameter(start['weights'][0].clone())
    values = start['weights'][1]
    optimizer = torch.optim.AdamW([weights], lr=args.lr, weight_decay=0)
    fit, held = train_cases()
    held = [Example(h, case, 'cpu') for case in held[:args.validation]]
    log = (args.out / 'log.jsonl').open('a')

    def validate(step):
        with torch.no_grad():
            value = sum(loss(h, e, weights, values).item() for e in held) / len(held)
        log.write(json.dumps({'step': step, 'validation': value, 'time': time.time()}) + '\n')
        log.flush()
        print(f'step {step} validation attention KL {value:.4f}', flush=True)
        return value

    best = validate(0)
    running = []
    for step, case in enumerate(fit[:args.steps], 1):
        value = loss(h, Example(h, case, h.device), weights, values)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_([weights], 1.0, error_if_nonfinite=True)
        optimizer.step()
        running.append(value.item())
        if step % args.every == 0 or step == args.steps:
            print(f'step {step} train attention KL {sum(running) / len(running):.4f}', flush=True)
            running.clear()
            current = validate(step)
            if current < best:
                best = current
                torch.save({'weights': torch.stack((weights.detach(), values)).cpu(),
                            'layers': start['layers'].cpu(), 'start': args.start, 'steps': step,
                            'validation': current}, args.out / 'keyfit.pt')
    print(f'best validation attention KL {best:.4f}')


if __name__ == '__main__':
    main()
