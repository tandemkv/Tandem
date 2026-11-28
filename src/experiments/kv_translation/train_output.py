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
--skew fits the answer span mode-seeking: 0.3 forward + 0.7 reverse KL, each answer position weighted by
1 + 4 JSD(native with the prefix || native without it), plus 0.25 forward KL on the suffix positions.
--bridge mixes in HotpotQA train prompts in the bridge format; --accum and --warmup average gradients.
--probes holds out bridge prompts and checks greedy answers on them at each validation: entity match with
native and top-1 on passage-1-dependent answer tokens. The best checkpoint is then picked by entity match.
--readout adds the per-layer readout loss: under native queries, each layer's attention output over the
translated prefix and native suffix should match the native one (normalized squared error, layer mean).

  train_output.py --start keyfit --steps 4000 --out DIR   -> DIR/output.pt, DIR/log.jsonl
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

import harness
from conditions import Prepared
from data import bridge_train_cases
from mapper import capture, target_embeddings, to_cache, translate
from metrics import entities
from run import git_commit
from train_keys import VALIDATION, attention, native_queries, rotate, train_cases, unrotated_keys


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
    def __init__(self, h, case, device, header=0, answer_only=False, readout=False, skew=False):
        p = Prepared(h, case)
        native, _ = h.generate(p.target_ids + p.suffix)
        self.ids = p.suffix + native['tokens'][:-1]
        self.native = native['answer']
        self.alignments = {'align': p.alignment, 'overlap': p.overlap}
        self.target = p.target_ids
        self.answer = len(p.suffix) - 1  # The first teacher position that predicts an answer token.
        teacher = logits(h, p.target_ids + self.ids)[-len(self.ids):].float().log_softmax(-1)
        if answer_only:  # Keep the positions that predict the answer, from the last question token on.
            teacher = teacher[len(p.suffix) - 1:]
        self.header = capture(h.qwen, p.target_ids[:header]).to(device) if header else None
        self.tensors = [t.to(device) for t in (p.source, teacher)]
        self.weights = None
        if skew:  # How much each answer position depends on the prefix, under native Qwen.
            base = logits(h, self.ids)[self.answer:].float().log_softmax(-1)
            middle = torch.logaddexp(teacher[self.answer:], base) - torch.log(torch.tensor(2.0))
            jsd = 0.5 * ((teacher[self.answer:].exp() * (teacher[self.answer:] - middle)).sum(-1)
                         + (base.exp() * (base - middle)).sum(-1))
            self.weights = (1 + 4 * jsd).to(device)
        self.readout = None
        if readout:  # Native suffix queries, and native rotated keys and values, per layer.
            full = p.target_ids + self.ids
            queries, _ = native_queries(h, full)
            native = capture(h.qwen, full)
            self.prefix = len(p.target_ids)
            self.readout = [t.to(device) for t in (queries[:, :, self.prefix:], rotate(h, unrotated_keys(h, native[:, 0])),
                                                   unrotated_keys(h, native[:, 1]))]


def readout_error(h, example, features):
    queries, keys, values = example.readout

    def output(k, v):
        weights = attention(queries, k).exp()
        return weights @ v.repeat_interleave(queries.shape[1] // v.shape[1], dim=1)

    target = output(keys, values)
    k = torch.cat((rotate(h, unrotated_keys(h, features[:, 0].float())), keys[:, :, example.prefix:]), dim=2)
    v = torch.cat((unrotated_keys(h, features[:, 1].float()), values[:, :, example.prefix:]), dim=2)
    return (((output(k, v) - target) ** 2).mean((1, 2, 3)) / (target ** 2).mean((1, 2, 3))).mean()


def embed(h, example, mapping):
    return target_embeddings(h.qwen, example.target) if 'mix_emb' in mapping else None


@torch.no_grad()
def probe(h, probes, mapping):
    """Greedy entity match with native, and top-1 on answer positions that depend on passage 1 (w >= 1.5)."""
    matched, hits, total = 0, 0, 0
    for example in probes:
        source, teacher = (t.to(h.device) for t in example.tensors)
        features = translate(source, example.alignments[mapping.get('pooling', 'align')], mapping, h.normalization,
                             h.smol.config.num_key_value_heads, embed(h, example, mapping))
        student = logits(h, example.ids, to_cache(h.qwen, features))[example.answer:]
        needed = example.weights.to(h.device) >= 1.5
        hits += (student.argmax(-1) == teacher[example.answer:].argmax(-1))[needed].sum().item()
        total += needed.sum().item()
        answer, _ = h.generate(example.ids[:example.answer + 1], to_cache(h.qwen, features))
        matched += entities(answer['answer']) == entities(example.native)
    return matched / len(probes), hits / max(total, 1)


def loss(h, example, mapping, readout=0.0, skew=False):
    source, teacher = (t.to(h.device) for t in example.tensors)
    alignment = example.alignments[mapping.get('pooling', 'align')]
    features = translate(source, alignment, mapping, h.normalization, h.smol.config.num_key_value_heads,
                         embed(h, example, mapping))
    if example.header is not None:  # The first prefix tokens come from Qwen itself, as in native_header.
        features = torch.cat([example.header.to(features), features[len(example.header):]])
    student = logits(h, example.ids, to_cache(h.qwen, features))[-len(teacher):].float().log_softmax(-1)
    forward = (teacher.exp() * (teacher - student)).sum(-1)
    if skew:
        reverse = (student.exp() * (student - teacher)).sum(-1)[example.answer:]
        w = example.weights.to(h.device)
        kl = (0.25 * forward[:example.answer].mean() if example.answer else 0.0) + (
            w * (0.3 * forward[example.answer:] + 0.7 * reverse)).sum() / w.sum()
    else:
        kl = forward.mean()
    return kl + readout * readout_error(h, example, features) if readout else kl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', choices=tuple(harness.CHECKPOINTS), default='keyfit')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--offset', type=int, default=0, help='Skip this many training prompts, to continue a run')
    parser.add_argument('--header', type=int, default=0, help='Train with this many native leading prefix tokens')
    parser.add_argument('--coupled', action='store_true', help='Let K and V each read both source K and V')
    parser.add_argument('--answer-only', action='store_true', help='Fit only the positions that predict the answer')
    parser.add_argument('--mlp', type=int, default=0, help='Add a nonlinear residual with this hidden width')
    parser.add_argument('--mlp-lr', type=float, default=1e-4)
    parser.add_argument('--readout', type=float, default=0.0, help='Weight of the per-layer readout loss')
    parser.add_argument('--pooling', choices=('align', 'overlap'), help='Source pooling; default: the start map\'s')
    parser.add_argument('--skew', action='store_true', help='Mode-seeking, prefix-weighted answer loss')
    parser.add_argument('--bridge', type=float, default=0.0, help='Share of prompts in the bridge format')
    parser.add_argument('--accum', type=int, default=1, help='Prompts per optimizer step')
    parser.add_argument('--warmup', type=int, default=0, help='Optimizer steps of linear learning-rate warmup')
    parser.add_argument('--cosine', type=float, default=0.0, help='After warmup, decay each lr to this share of it')
    parser.add_argument('--probes', type=int, default=0, help='Held-out bridge prompts for greedy checks')
    parser.add_argument('--mix', type=int, default=0, help='Add a cross-token K/V residual with this bottleneck width')
    parser.add_argument('--mix-lr', type=float, default=3e-5)
    parser.add_argument('--embed', action='store_true', help='Let the mixing residual read the target token embedding')
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
    mapping = {'layers': layers, 'coupled': coupled, 'weights': torch.nn.Parameter(weights),
               'pooling': args.pooling or start.get('pooling', 'align')}
    groups = [{'params': [mapping['weights']], 'lr': args.lr}]
    if 'up' in start or args.mlp:  # Zero down weights: the residual map starts as the start map.
        up = start['up'].clone() if 'up' in start else torch.randn(*weights.shape[:-1], args.mlp, generator=torch.Generator().manual_seed(0)) / weights.shape[-2] ** 0.5
        down = start['down'].clone() if 'down' in start else torch.zeros(*weights.shape[:-2], up.shape[-1], weights.shape[-1])
        mapping |= {'up': torch.nn.Parameter(up.to(weights)), 'down': torch.nn.Parameter(down.to(weights))}
        groups.append({'params': [mapping['up'], mapping['down']], 'lr': args.mlp_lr})
    if 'mix_down' in start or args.mix:  # Zero up weights: the residual map starts as the start map.
        width = start['mix_down'].shape[-1] if 'mix_down' in start else args.mix
        layers_read = torch.arange(3, h.smol.config.num_hidden_layers, 4)  # 8 evenly spaced SmolLM layers.
        inputs = 2 * len(layers_read) * h.smol.config.num_key_value_heads * (
            h.smol.config.hidden_size // h.smol.config.num_attention_heads)
        target = h.qwen.config.num_hidden_layers
        fresh = {'mix_layers': layers_read,
                 'mix_down': torch.randn(target, inputs, width, generator=torch.Generator().manual_seed(0)) / inputs ** 0.5,
                 'mix_conv': torch.zeros(target, width, 5),
                 'mix_up': torch.zeros(target, width, weights.shape[-1] * 2)}
        if args.embed or 'mix_emb' in start:  # Zero-init: adds nothing until trained.
            fresh['mix_emb'] = torch.zeros(target, h.qwen.config.hidden_size, width)
        for name, value in fresh.items():
            value = start[name].clone() if name in start else value
            mapping[name] = value.to(weights.device) if name == 'mix_layers' else torch.nn.Parameter(value.to(weights))
        groups.append({'params': [mapping[k] for k in ('mix_down', 'mix_conv', 'mix_up', 'mix_emb') if k in mapping],
                       'lr': args.mix_lr})
    parameters = [p for group in groups for p in group['params']]
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    total = max(args.steps // args.accum, 1)

    def scale(n):
        warm = min(1.0, (n + 1) / args.warmup) if args.warmup else 1.0
        if not args.cosine:
            return warm
        done = min(1.0, max(0, n - args.warmup) / max(total - args.warmup, 1))
        return warm * (args.cosine + (1 - args.cosine) * 0.5 * (1 + math.cos(math.pi * done)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    fit, held = train_cases()
    if args.bridge:  # A fixed mix: each prompt is a bridge prompt with probability --bridge.
        bridge, plain = iter(bridge_train_cases()[args.offset:-args.probes or None]), iter(fit[args.offset:])
        choose = random.Random(20261127)
        fit = [next(bridge) if choose.random() < args.bridge else next(plain) for _ in range(args.steps)]
    else:
        fit = fit[args.offset:args.offset + args.steps]
    held = [Example(h, case, 'cpu', args.header, args.answer_only) for case in held[:args.validation]]
    probes = []
    for case in bridge_train_cases()[-args.probes:] if args.probes else []:
        try:
            probes.append(Example(h, case, 'cpu', skew=True))
        except ValueError:
            pass
    log = (args.out / 'log.jsonl').open('a')

    def validate(step):
        with torch.no_grad():
            value = sum(loss(h, e, mapping).item() for e in held) / len(held)
        record = {'step': step, 'validation': value, 'time': time.time()}
        if probes:
            record['probe_entities'], record['probe_p1_top1'] = probe(h, probes, mapping)
        log.write(json.dumps(record) + '\n')
        log.flush()
        print(f'step {step} validation output KL {value:.4f}' + (
            f" probe entities {record['probe_entities']:.3f} p1 top1 {record['probe_p1_top1']:.3f}" if probes else ''),
            flush=True)
        return -record['probe_entities'] if probes else value

    best = validate(0)
    running = []
    optimizer.zero_grad(set_to_none=True)
    for step, case in enumerate(fit, 1):
        try:
            example = Example(h, case, h.device, args.header, args.answer_only, args.readout > 0, args.skew)
        except ValueError as error:  # The prompt has no clean shared split for both tokenizers.
            print(f'skip {case.id}: {error}', flush=True)
            continue
        value = loss(h, example, mapping, args.readout, args.skew)
        (value / args.accum).backward()
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        running.append(value.item())
        if (step % args.every == 0 or step == args.steps) and running:
            print(f'step {step} train output KL {sum(running) / len(running):.4f}', flush=True)
            running.clear()
            current = validate(step)
            if current < best or probes:
                state = {k: v.detach().cpu() for k, v in mapping.items() if torch.is_tensor(v)} | {
                            'coupled': coupled, 'pooling': mapping['pooling'], 'skew': args.skew, 'bridge': args.bridge,
                            'accum': args.accum, 'warmup': args.warmup, 'cosine': args.cosine, 'mix_lr': args.mix_lr, 'mlp_lr': args.mlp_lr, 'start': args.start,
                            'offset': args.offset, 'header': args.header, 'answer_only': args.answer_only, 'readout': args.readout, 'steps': step, 'commit': git_commit(), 'lr': args.lr, 'validation': current}
                if probes:  # Keep every checkpoint: the probe and the validation KL can disagree.
                    torch.save(state, args.out / f'output-{step}.pt')
                if current < best:
                    best = current
                    torch.save(state, args.out / 'output.pt')
    print(f'best validation {abs(best):.4f}')


if __name__ == '__main__':
    main()
