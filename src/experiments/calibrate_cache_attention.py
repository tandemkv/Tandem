# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Calibrate the existing mapper on native receiver continuations, on one GPU."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

from requests.exceptions import RequestException
import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM
import paired_kv_dataset as base
from cache_attention import CacheAttention
from train_cache_attention import (Dataset, REPO, causal_positions, checkpoint,
                                   logit_kl, receiver_logits, translate_prefix, upload_outputs)
from evaluate_cache_attention import (CHECKPOINT, CHECKPOINT_REVISION, cached_start,
                                      check_destination, continue_greedy, lock_workdir,
                                      native_start, prefix_agreement)


def continuation_inputs(prompt, continuation):
    """Predict continuation[0] from the final prompt token, then each next token."""
    if not continuation:
        raise ValueError('Empty continuation')
    previous = torch.tensor([continuation[:-1]], dtype=prompt.dtype, device=prompt.device)
    return torch.cat([prompt, previous], -1), torch.cat([prompt[:, -1:], previous], -1)


def teacher_logits(receiver, prompt, continuation):
    full, _ = continuation_inputs(prompt, continuation)
    with torch.no_grad():
        return receiver(input_ids=full, use_cache=False,
                        logits_to_keep=len(continuation)).logits


def stopping_update(value, best, stale):
    if not math.isfinite(value):
        raise ValueError('Nonfinite validation loss')
    return (value, 0, True) if value < best else (best, stale + 1, False)


def restore_progress(out, mapper, optimizer, run, provenance, train_count, device):
    saved = torch.load(out / 'last.pt', map_location=device, weights_only=True)
    old = saved['calibration']
    identity = ('parent_revision', 'parent_checkpoint', 'parent_sha256', 'parent_run',
                'tokens', 'handoff_prompt_tokens', 'lr', 'seed', 'train_ids',
                'validation_ids', 'pilot', 'objective')
    if saved['run'] != run or any(old[k] != provenance[k] for k in identity):
        raise ValueError('Resume checkpoint configuration mismatch')
    mapper.load_state_dict(saved['model'])
    if 'training_state' in saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_state'].cpu())
        if 'cuda_rng_state' in saved:
            torch.cuda.set_rng_state(saved['cuda_rng_state'].cpu())
        return saved['training_state'], False
    # V1 saved weights and update counts, but omitted Adam moments. Continue
    # weights with a fresh optimizer and preserve that limitation in provenance.
    steps = saved['steps']
    epoch, position = divmod(steps, train_count)
    best = json.loads((out / 'before.json').read_text())['summary']['kl']
    stale = 0
    metrics = out / 'metrics.jsonl'
    for row in (json.loads(line) for line in (metrics.read_text().splitlines() if metrics.exists() else []) if line.strip()):
        if row['steps'] <= steps and row['prompts_seen_this_epoch'] == train_count:
            best, stale, _ = stopping_update(row['validation']['kl'], best, stale)
    return {'epoch': epoch, 'position': position, 'steps': steps, 'best': best,
            'stale': stale, 'epoch_loss_sum': 0., 'epoch_loss_count': 0}, True


def save_progress(out, mapper, optimizer, run, provenance, state):
    payload = {'model': mapper.state_dict(), 'run': run, 'calibration': provenance,
               'steps': state['steps'], 'optimizer': optimizer.state_dict(),
               'training_state': dict(state), 'rng_state': torch.get_rng_state()}
    if next(mapper.parameters()).is_cuda:
        payload['cuda_rng_state'] = torch.cuda.get_rng_state()
    checkpoint(out / 'last.pt', payload)


class CheckpointPublisher:
    """Keep local training independent of transient Hub upload failures."""
    def __init__(self, args, upload=upload_outputs, clock=time.time, interval=600):
        self.args, self.upload, self.clock, self.interval = args, upload, clock, interval
        self.pending = args.work / 'upload-pending.json'
        self.next_upload, self.blocked_until = 0., 0.
        if self.pending.exists():
            self.blocked_until = json.loads(self.pending.read_text())['retry_after_unix']

    def __call__(self, final=False):
        now = self.clock()
        if now < self.blocked_until or (not final and now < self.next_upload):
            return False
        try:
            self.upload(self.args)
        except RequestException as error:
            response = getattr(error, 'response', None)
            status = response.status_code if response is not None else None
            if status is not None and status not in (408, 429, 500, 502, 503, 504):
                raise
            delay = 3600 if status == 429 else self.interval
            if response is not None:
                retry = response.headers.get('Retry-After', '')
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    try:
                        delay = max(delay, parsedate_to_datetime(retry).timestamp() - now)
                    except (TypeError, ValueError, OverflowError):
                        pass
            self.blocked_until = now + delay
            base.write_json(self.pending, {'retry_after_unix': self.blocked_until,
                            'status': status, 'outputs': str(self.args.work / 'outputs')})
            print(f'UPLOAD DEFERRED: HTTP {status}; local checkpoints retained; retry in {delay:.0f}s. Training continues.', flush=True)
            return False
        self.next_upload = now + self.interval
        self.blocked_until = 0.
        self.pending.unlink(missing_ok=True)
        return True


def score(mapper, receiver, data, samples, traces, args, greedy=False):
    mapper.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            source = data.features(sample, args.source)
            target = data.features(sample, args.target)
            teacher = teacher_logits(receiver, ids, traces[sample['id']])
            _, suffix = continuation_inputs(ids, traces[sample['id']])
            control = receiver_logits(receiver, target[:-1].to(args.device), suffix)
            if not torch.allclose(control, teacher, rtol=1e-3, atol=1e-3):
                raise ValueError('Native continuation replay control failed')
            mapped = translate_prefix(mapper, source, sample, args, ids.shape[1] - 1)
            student = receiver_logits(receiver, mapped, suffix)
            row = {'id': sample['id'], 'kl': logit_kl(student, teacher).item(),
                   'teacher_path_top1': (student.argmax(-1) == teacher.argmax(-1)).float().mean().item(),
                   'control_max_logit_error': (control - teacher).abs().max().item()}
            if not torch.isfinite(torch.tensor(row['kl'])):
                raise ValueError('Nonfinite continuation evaluation loss')
            if greedy:
                output = cached_start(receiver, mapped, ids[:, -1:])
                generated = continue_greedy(receiver, output, args.tokens, args.eos_ids)
                row.update(prefix_agreement(traces[sample['id']], generated))
                row['native_tokens'], row['mapped_tokens'] = traces[sample['id']], generated
            rows.append(row)
    summary = {key: sum(r[key] for r in rows) / len(rows) for key in ('kl', 'teacher_path_top1')}
    if greedy:
        summary.update({key: sum(r[key] for r in rows) / len(rows)
                        for key in ('first_token_match', 'matching_prefix_tokens', 'sequence_equal')})
    return {'summary': summary, 'samples': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--pilot', action='store_true', help='Eight train / four validation prompts')
    p.add_argument('--epochs', type=int, default=30, help='Total passes, including completed passes when resuming')
    p.add_argument('--patience', type=int, default=5, help='Stop after this many complete passes without validation KL improvement')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--tokens', type=int, default=32, help='Use 1 for the matched one-token calibration control')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--hours', type=float, default=2)
    p.add_argument('--upload-prefix', required=True)
    args = p.parse_args()
    args.device = 'cuda'
    if min(args.epochs, args.tokens, args.lr, args.hours, args.patience) <= 0:
        p.error('Training budgets must be positive')
    if args.tokens > 32:
        p.error('This bounded experiment supports at most 32 continuation tokens')
    if not args.upload_prefix.startswith('experiments/') or '..' in args.upload_prefix:
        p.error('Use a unique experiments/... destination')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error('Set CUDA_VISIBLE_DEVICES to one allocated GPU')
    args.work.mkdir(parents=True, exist_ok=True)
    lock = lock_workdir(args.work)
    out = args.work / 'outputs'
    if out.exists() and not args.resume:
        p.error('Work directory already contains outputs; use a new directory')
    if args.resume and not all((out / name).exists() for name in
                              ('last.pt', 'best.pt', 'run.json', 'before.json', 'train-before.json', 'native-traces.json')):
        p.error('Resume requires an existing calibration checkpoint and baseline outputs')
    api = HfApi()
    publish = CheckpointPublisher(args)
    if args.resume:
        if not api.dataset_info(REPO).private:
            p.error('Resume destination must be private')
        receipt_path = args.work / 'upload-receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['prefix'] != args.upload_prefix:
                p.error('Resume must use the original upload prefix')
        else:
            check_destination(api, args.upload_prefix)  # Initial commit may have failed before any receipt.
    else:
        check_destination(api, args.upload_prefix)
    info = api.dataset_info(REPO, revision=CHECKPOINT_REVISION, files_metadata=True)
    if not info.private:
        raise ValueError('Expected private parent checkpoint')
    entry = next(s for s in info.siblings if s.rfilename == CHECKPOINT)
    path = Path(hf_hub_download(REPO, CHECKPOINT, repo_type='dataset', revision=CHECKPOINT_REVISION))
    if not entry.lfs or base.file_sha(path) != entry.lfs.sha256:
        raise ValueError('Parent checkpoint checksum mismatch')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    run = parent['run']
    for key, filename in [('training_script_sha256', 'train_cache_attention.py'),
                          ('architecture_script_sha256', 'cache_attention.py')]:
        if run[key] != base.file_sha(Path(__file__).with_name(filename)):
            raise ValueError('Parent checkpoint code identity mismatch')
    args.source, args.target, args.batch = run['source'], run['target'], run['batch']
    data = Dataset(SimpleNamespace(local_data=None, data_root=args.data_root, pilot=False))
    if (data.manifest_sha != run['manifest_sha256'] or
            [s['id'] for s in data.train] != run['train_ids'] or
            [s['id'] for s in data.validation] != run['validation_ids']):
        raise ValueError('Parent checkpoint dataset mismatch')
    train = data.train[:8] if args.pilot else data.train
    validation = data.validation[:4] if args.pilot else data.validation
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    mapper = CacheAttention(run['config']).to(args.device)
    mapper.load_state_dict(parent['model'])
    spec = base.MODELS[args.target]
    receiver = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
                dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
    receiver.requires_grad_(False)
    eos = receiver.generation_config.eos_token_id
    args.eos_ids = set(eos if isinstance(eos, list) else [eos])
    out.mkdir(exist_ok=args.resume)
    provenance = {'parent_revision': CHECKPOINT_REVISION, 'parent_checkpoint': CHECKPOINT,
        'parent_sha256': entry.lfs.sha256, 'parent_run': run, 'tokens': args.tokens,
        'handoff_prompt_tokens': 1, 'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
        'train_ids': [s['id'] for s in train], 'validation_ids': [s['id'] for s in validation],
        'script_sha256': base.file_sha(Path(__file__)), 'pilot': args.pilot,
        'objective': 'native-continuation teacher-to-student KL + 0.01 standardized prefix KV MSE',
        'evaluation_scope': 'Development validation already used for parent selection; not a final test'}
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    state = None
    if args.resume:
        state, reset_optimizer = restore_progress(out, mapper, optimizer, run, provenance, len(train), args.device)
        provenance['resume'] = {'from_steps': state['steps'], 'optimizer_reset': reset_optimizer,
                                'previous_run': json.loads((out / 'run.json').read_text())}
        provenance['patience'] = args.patience
        base.write_json(out / f'resume-{time.time_ns()}.json', provenance)
        print(f"RESUME: {state['steps']} updates, {state['epoch']} complete passes; optimizer_reset={reset_optimizer}", flush=True)
    else:
        provenance['patience'] = args.patience
        base.write_json(out / 'run.json', provenance)
    traces = json.loads((out / 'native-traces.json').read_text()) if args.resume else {}
    expected_ids = {s['id'] for s in train + validation}
    if args.resume and (set(traces) != expected_ids or any(not 0 < len(t) <= args.tokens for t in traces.values())):
        raise ValueError('Saved continuation traces do not match this run')
    with torch.no_grad():
        for sample in ([] if args.resume else train + validation):
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            native = native_start(receiver, ids)
            control = cached_start(receiver, data.features(sample, args.target)[:-1].to(args.device), ids[:, -1:])
            if not torch.allclose(native.logits, control.logits, rtol=1e-3, atol=1e-3):
                raise ValueError('Native one-token replay control failed')
            trace = continue_greedy(receiver, native, args.tokens, args.eos_ids)
            if trace != continue_greedy(receiver, control, args.tokens, args.eos_ids):
                raise ValueError('Native continuation control failed')
            traces[sample['id']] = trace
    if args.resume:
        before = json.loads((out / 'before.json').read_text())
        train_before = json.loads((out / 'train-before.json').read_text())
        score(mapper, receiver, data, validation, traces, args)  # Recheck native replay on this device.
    else:
        base.write_json(out / 'native-traces.json', traces)
        before = score(mapper, receiver, data, validation, traces, args, greedy=True)
        base.write_json(out / 'before.json', before)
        train_before = score(mapper, receiver, data, train[:8], traces, args, greedy=True)
        base.write_json(out / 'train-before.json', train_before)
        checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run, 'calibration': provenance})
        state = {'epoch': 0, 'position': 0, 'steps': 0, 'best': before['summary']['kl'],
                 'stale': 0, 'epoch_loss_sum': 0., 'epoch_loss_count': 0}
        save_progress(out, mapper, optimizer, run, provenance, state)
    print('BEFORE: '+json.dumps(before['summary']), flush=True)
    publish()
    stopped = False
    for epoch in range(state['epoch'], args.epochs):
        if state['stale'] >= args.patience:
            break
        order = list(train)
        random.Random(args.seed + epoch).shuffle(order)
        for position in range(state['position'], len(order)):
            if time.perf_counter() - started >= args.hours * 3600:
                stopped = True
                break
            sample = order[position]
            mapper.train()
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            source, target = data.features(sample, args.source), data.features(sample, args.target)
            teacher = teacher_logits(receiver, ids, traces[sample['id']])
            _, suffix = continuation_inputs(ids, traces[sample['id']])
            optimizer.zero_grad(set_to_none=True)
            mapped = translate_prefix(mapper, source, sample, args, ids.shape[1] - 1)
            student = receiver_logits(receiver, mapped, suffix)
            _, valid = causal_positions(sample, args.source, args.target)
            mask = valid[:-1].to(args.device)
            anchor = F.mse_loss(mapper.standardize_target(mapped)[mask],
                               mapper.standardize_target(target[:-1].to(args.device))[mask])
            loss = logit_kl(student, teacher) + .01 * anchor
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite calibration loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state['epoch_loss_sum'] += loss.item()
            state['epoch_loss_count'] += 1
            state['steps'] += 1
            state['position'] = position + 1
            if state['steps'] % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, prompt {position+1}/{len(train)}: {loss.item():.5f}", flush=True)
            if state['steps'] % 25 == 0:
                save_progress(out, mapper, optimizer, run, provenance, state)
                publish()
        if stopped:
            save_progress(out, mapper, optimizer, run, provenance, state)
            publish()
            break
        validation_score = score(mapper, receiver, data, validation, traces, args)
        state['best'], state['stale'], improved = stopping_update(validation_score['summary']['kl'], state['best'], state['stale'])
        if improved:
            checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run, 'calibration': provenance})
        row = {'epoch': epoch+1, 'steps': state['steps'], 'prompts_seen_this_epoch': len(train),
               'training_loss': state['epoch_loss_sum']/state['epoch_loss_count'] if state['epoch_loss_count'] else None,
               'validation': validation_score['summary'], 'stale_epochs': state['stale'],
               'elapsed_seconds': time.perf_counter()-started}
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        state.update(epoch=epoch+1, position=0, epoch_loss_sum=0., epoch_loss_count=0)
        save_progress(out, mapper, optimizer, run, provenance, state)
        publish()
    mapper.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True)['model'])
    after = score(mapper, receiver, data, validation, traces, args, greedy=True)
    train_after = score(mapper, receiver, data, train[:8], traces, args, greedy=True)
    result = {'before': before, 'after': after, 'train_before': train_before,
              'train_after': train_after, 'steps': state['steps'], 'budget_reached': stopped,
              'complete_epochs': state['epoch'], 'early_stopped': state['stale'] >= args.patience,
              'patience': args.patience, 'epochs_requested': args.epochs,
              'wall_seconds': time.perf_counter()-started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
    base.write_json(out / 'comparison.json', result)
    uploaded = publish(final=True)
    if not uploaded:
        print(f'UPLOAD PENDING: completed outputs are saved at {out}', flush=True)
    print('COMPLETE: '+json.dumps({'before': before['summary'], 'after': after['summary']}), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
