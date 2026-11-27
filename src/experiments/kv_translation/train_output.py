# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "accelerate==1.9.0", "bitsandbytes==0.50.2",
#   "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0",
# ]
# ///
"""Refit both halves of a map end to end: Qwen after the translated cache should predict like native Qwen.

Unlike train_keys.py, the loss runs through every Qwen layer, so errors that compound across layers count.
The loss is KL from native to translated next-token distributions over the question and native answer.
Qwen stays frozen; gradients reach the map through the translated cache.

  train_output.py --start keyfit --steps 4000 --out DIR   -> DIR/output.pt, DIR/log.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch

import harness
from conditions import Prepared
from mapper import capture, to_cache, translate
from run import git_commit
from train_keys import VALIDATION, train_cases


def logits(h, ids, cache=None):
    """Qwen logits for every position of ids, after an optional cache."""
    past = cache.get_seq_length() if cache is not None else 0
    positions = torch.arange(past, past + len(ids), device=h.device)
    return h.qwen(input_ids=torch.tensor([ids], device=h.device), past_key_values=cache, use_cache=True,
                  cache_position=positions, position_ids=positions[None],
                  attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long, device=h.device)).logits[0]


class Example:
    """Source K/V and native log-probabilities over the question and native answer for one prompt."""

    @torch.no_grad()
    def __init__(self, h, case, device, header=0):
        p = Prepared(h, case)
        native, _ = h.generate(p.target_ids + p.suffix)
        self.ids = p.suffix + native['tokens'][:-1]
        self.alignment = p.alignment
        teacher = logits(h, p.target_ids + self.ids)[-len(self.ids):].float().log_softmax(-1)
        self.header = capture(h.qwen, p.target_ids[:header]).to(device) if header else None
        self.tensors = [t.to(device) for t in (p.source, teacher)]


def loss(h, example, weights, layers, coupled=False):
    source, teacher = (t.to(h.device) for t in example.tensors)
    features = translate(source, example.alignment, {'layers': layers, 'weights': weights, 'coupled': coupled}, h.normalization,
                         h.smol.config.num_key_value_heads)
    if example.header is not None:  # The first prefix tokens come from Qwen itself, as in native_header.
        features = torch.cat([example.header.to(features), features[len(example.header):]])
    student = logits(h, example.ids, to_cache(h.qwen, features)).float().log_softmax(-1)
    return (teacher.exp() * (teacher - student)).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', choices=tuple(harness.CHECKPOINTS), default='keyfit')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--offset', type=int, default=0, help='Skip this many training prompts, to continue a run')
    parser.add_argument('--header', type=int, default=0, help='Train with this many native leading prefix tokens')
    parser.add_argument('--coupled', action='store_true', help='Let K and V each read both source K and V')
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
    weights, layers, coupled = start['weights'].clone(), start['layers'], args.coupled or start.get('coupled', False)
    if coupled and not start.get('coupled'):  # Zero cross weights: the coupled map starts as the start map.
        weights = torch.cat((weights[..., :-1, :], torch.zeros_like(weights[..., :-1, :]), weights[..., -1:, :]), -2)
    weights = torch.nn.Parameter(weights)
    optimizer = torch.optim.AdamW([weights], lr=args.lr, weight_decay=0)
    fit, held = train_cases()
    held = [Example(h, case, 'cpu', args.header) for case in held[:args.validation]]
    log = (args.out / 'log.jsonl').open('a')

    def validate(step):
        with torch.no_grad():
            value = sum(loss(h, e, weights, layers, coupled).item() for e in held) / len(held)
        log.write(json.dumps({'step': step, 'validation': value, 'time': time.time()}) + '\n')
        log.flush()
        print(f'step {step} validation output KL {value:.4f}', flush=True)
        return value

    best = validate(0)
    running = []
    for step, case in enumerate(fit[args.offset:args.offset + args.steps], 1):
        try:
            example = Example(h, case, h.device, args.header)
        except ValueError as error:  # The prompt has no clean shared split for both tokenizers.
            print(f'skip {case.id}: {error}', flush=True)
            continue
        value = loss(h, example, weights, layers, coupled)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_([weights], 1.0, error_if_nonfinite=True)
        optimizer.step()
        running.append(value.item())
        if (step % args.every == 0 or step == args.steps) and running:
            print(f'step {step} train output KL {sum(running) / len(running):.4f}', flush=True)
            running.clear()
            current = validate(step)
            if current < best:
                best = current
                torch.save({'weights': weights.detach().cpu(), 'layers': layers.cpu(), 'start': args.start,
                            'offset': args.offset, 'header': args.header, 'coupled': coupled, 'steps': step, 'commit': git_commit(), 'lr': args.lr, 'validation': current},
                           args.out / 'output.pt')
    print(f'best validation output KL {best:.4f}')


if __name__ == '__main__':
    main()
