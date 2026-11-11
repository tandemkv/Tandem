# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Train and evaluate one-direction conditional cache attention on existing shards."""
import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import paired_kv_dataset as base
import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, DynamicCache
from cache_attention import CacheAttention

REPO = 'tandemkv/paired-kv-qwen-smollm'
REVISION = 'dca861cd61b10859d5682565cfbba5606e5c5974'
PREFIX = 'runs/hotpot-200-20261028'
MANIFEST_SHA = 'b88a13e01fd31f9fe56e2dd05b10e6703279080d3669f033c131ecfcdb0b9532'


def stack_features(tensors, geometry):
    return torch.stack([torch.stack([tensors[f'layer_{i}.{role}'].reshape(-1, geometry['kv_heads'], geometry['head_dim'])
        for i in range(geometry['layers'])], dim=1).flatten(1, 2) for role in ('k', 'v')], dim=1)


def causal_positions(sample, source, target):
    aligned = sample['alignment'][f'{source}_to_{target}']['target_to_source']
    ends = sample['tokens'][source]['offsets']
    target_ends = sample['tokens'][target]['offsets']
    valid = torch.tensor([ends[i][1] <= target_ends[j][1] and target_ends[j][1] > target_ends[j][0]
                          for j, i in enumerate(aligned)], dtype=torch.bool)
    return torch.tensor(aligned), valid


class Dataset:
    def __init__(self, args):
        self.args, self.checked = args, {}
        self.remote = args.local_data is None
        if self.remote:
            info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
            if not info.private:
                raise ValueError('Expected a private dataset')
            self.root = args.data_root / PREFIX
            manifest_path = self.download('manifest.json')
            checkpoint = json.loads(self.download('checkpoint.json').read_text())
            if base.file_sha(manifest_path) != MANIFEST_SHA or checkpoint['manifest_sha256'] != MANIFEST_SHA:
                raise ValueError('Pinned dataset checksum mismatch')
            self.records = checkpoint['shards']
            for name, record in self.records.items():
                entry = next(s for s in info.siblings if s.rfilename == f'{PREFIX}/{name}')
                if entry.lfs is None or entry.lfs.sha256 != record['sha256']:
                    raise ValueError('Remote shard receipt mismatch')
        else:
            self.root = args.local_data
            manifest_path = self.root / 'manifest.json'
            self.records = None
        self.manifest_sha = base.file_sha(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        base.validate_manifest(self.manifest)
        self.train = [s for s in self.manifest['samples'] if s['split'] == 'train']
        self.validation = [s for s in self.manifest['samples'] if s['split'] == 'validation']
        if args.pilot:
            self.train, self.validation = self.train[:4], self.validation[:2]
        if not self.train or not self.validation:
            raise ValueError('Need separate training and validation prompts')

    def download(self, name):
        return Path(hf_hub_download(REPO, f'{PREFIX}/{name}', repo_type='dataset', revision=REVISION,
                                   local_dir=self.args.data_root))

    def features(self, sample, alias):
        name = f'{alias}/{sample["id"]}.safetensors'
        path = self.root / name
        if not path.exists() and self.remote:
            path = self.download(name)
        record = self.records[name] if self.remote else json.loads(path.with_suffix('.json').read_text())
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.checked.get(path) != signature:
            expected_hash = record['sha256'] if self.remote else record['file_sha256']
            if base.file_sha(path) != expected_hash:
                raise ValueError(f'Shard checksum mismatch: {name}')
            with safe_open(str(path), framework='pt') as f:
                identity = json.loads(f.metadata()['identity'])
            expected = {'model': base.MODELS[alias], 'sample': sample['id'], 'text_sha256': sample['text_sha256'],
                        'manifest_sha256': self.manifest_sha, 'feature_space': base.SPACE}
            if identity != expected:
                raise ValueError('Shard provenance mismatch')
            self.checked[path] = signature
        tensors = load_file(str(path))
        shape = dict(base.EXPECTED_GEOMETRY[alias], tokens=len(sample['tokens'][alias]['ids']))
        base.validate_features(tensors, shape)
        return stack_features(tensors, shape)

    def pair(self, sample, stride=1):
        x, y = self.features(sample, self.args.source), self.features(sample, self.args.target)
        aligned, valid = causal_positions(sample, self.args.source, self.args.target)
        positions = valid.nonzero().flatten()[::stride]
        if not len(positions):
            raise ValueError('No causal aligned positions')
        return x[aligned[positions]], y[positions]


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_statistics_and_ridge(model, data, args):
    """Train-only float64 moments and sufficient statistics, one prompt at a time."""
    device = args.device
    sums = [torch.zeros_like(model.source_mean, dtype=torch.float64, device='cpu'), torch.zeros_like(model.target_mean, dtype=torch.float64, device='cpu')]
    squares = [s.clone() for s in sums]
    count = 0
    for sample in data.train:
        pair = data.pair(sample, args.stride)
        count += len(pair[0])
        for i, values in enumerate(pair):
            values = values.double()
            sums[i] += values.sum(0)
            squares[i] += values.square().sum(0)
    if count < 2:
        raise ValueError('Insufficient training positions')
    for i, (mean_name, std_name) in enumerate([('source_mean', 'source_std'), ('target_mean', 'target_std')]):
        mean = sums[i] / count
        std = (squares[i] / count - mean.square()).clamp_min(1e-6).sqrt()
        getattr(model, mean_name).copy_(mean.to(device=device, dtype=torch.float32))
        getattr(model, std_name).copy_(std.to(device=device, dtype=torch.float32))
    c = model.config
    dim = c['source_heads'] * c['head_dim'] + 1
    xx = torch.zeros(2, c['target_layers'], dim, dim, dtype=torch.float64, device=device)
    xy = torch.zeros(2, c['target_layers'], dim, c['target_heads'] * c['head_dim'], dtype=torch.float64, device=device)
    for sample in data.train:
        x, y = [v.to(device) for v in data.pair(sample, args.stride)]
        x = model.standardize_source(x).reshape(-1, 2, c['source_layers'], dim - 1)[:, :, model.layer_map]
        x = torch.cat([x, torch.ones_like(x[..., :1])], -1).double()
        y = model.standardize_target(y).reshape(-1, 2, c['target_layers'], c['target_heads'] * c['head_dim']).double()
        xx += torch.einsum('brli,brlj->rlij', x, x)
        xy += torch.einsum('brli,brlj->rlij', x, y)
    regularizer = torch.eye(dim, device=device, dtype=torch.float64) * args.ridge
    regularizer[-1, -1] = 0  # Intercept is not penalized.
    model.ridge.copy_(torch.linalg.solve(xx / count + regularizer, xy / count).float())
    return count


def validation_mse(model, data, args):
    totals, n = [0.0, 0.0], 0
    model.eval()
    with torch.no_grad():
        for sample in data.validation:
            x, y = data.pair(sample, args.stride)
            for start in range(0, len(x), args.batch):
                xb, yb = x[start:start + args.batch].to(args.device), y[start:start + args.batch].to(args.device)
                yb = model.standardize_target(yb)
                for i, affine in enumerate((True, False)):
                    totals[i] += F.mse_loss(model(xb, affine), yb, reduction='sum').item()
                n += yb.numel()
    return {'affine': totals[0] / n, 'attention': totals[1] / n}


def translate_prefix(mapper, source, sample, args, length, affine=False):
    alignment, valid = causal_positions(sample, args.source, args.target)
    # No preceding source boundary: use a train-only mean prior, never future KV.
    output = mapper.target_mean.unsqueeze(0).expand(length, -1, -1, -1).clone()
    positions = valid[:length].nonzero().flatten()
    for start in range(0, len(positions), args.batch):
        ids = positions[start:start + args.batch]
        values = mapper.predict(source[alignment[ids]].to(args.device), affine)
        output[ids.to(args.device)] = values
    return output


def to_cache(features, receiver):
    """Convert token-major pre-RoPE K and projected V to the receiver's cache."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
    apply_rope = qwen_rope if receiver.config.model_type == 'qwen2' else llama_rope
    g = base.geometry(receiver.config)
    length = len(features)
    positions = torch.arange(length, device=features.device)[None]
    cos, sin = receiver.model.rotary_emb(features, positions)
    cache = DynamicCache(config=receiver.config)
    shaped = features.reshape(length, 2, g['layers'], g['kv_heads'], g['head_dim'])
    for layer in range(g['layers']):
        key = shaped[:, 0, layer].permute(1, 0, 2).unsqueeze(0).contiguous()
        value = shaped[:, 1, layer].permute(1, 0, 2).unsqueeze(0).contiguous()
        _, key = apply_rope(key, key, cos, sin)
        cache.update(key, value, layer)
    return cache


def receiver_logits(receiver, features, suffix):
    cache = to_cache(features, receiver)
    positions = torch.arange(len(features), len(features) + suffix.shape[1], device=suffix.device)
    return receiver(input_ids=suffix, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, len(features) + suffix.shape[1], device=suffix.device, dtype=torch.long),
                    position_ids=positions[None], cache_position=positions).logits


def logit_kl(student, teacher):
    return F.kl_div(student.float().log_softmax(-1).flatten(0, 1),
                    teacher.float().log_softmax(-1).flatten(0, 1), log_target=True, reduction='batchmean')


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def behavior_evaluation(mapper, receiver, data, args):
    """Teacher-forced suffix behavior, with a same-path native cache control."""
    mapper.eval()
    rows = []
    with torch.no_grad():
        for sample in data.validation:
            source = data.features(sample, args.source)
            target = data.features(sample, args.target)
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            prefix_length = ids.shape[1] - args.probes
            if prefix_length < 1:
                raise ValueError('Probe suffix too long')
            synchronize(args.device)
            start = time.perf_counter()
            teacher = receiver(input_ids=ids, use_cache=False, logits_to_keep=args.probes).logits
            synchronize(args.device)
            native_ms = (time.perf_counter() - start) * 1000
            control = receiver_logits(receiver, target[:prefix_length].to(args.device), ids[:, -args.probes:])
            if not torch.allclose(control, teacher, rtol=1e-3, atol=1e-3):
                raise ValueError('Native cache replay control failed; no translation claims are valid')
            row = {'id': sample['id'], 'native_prefill_and_suffix_ms': native_ms,
                   'control_max_logit_error': (control - teacher).abs().max().item(), 'methods': {}}
            for affine, method in [(True, 'affine'), (False, 'attention')]:
                synchronize(args.device)
                start = time.perf_counter()
                features = translate_prefix(mapper, source, sample, args, prefix_length, affine)
                student = receiver_logits(receiver, features, ids[:, -args.probes:])
                synchronize(args.device)
                row['methods'][method] = {'kl': logit_kl(student, teacher).item(),
                    'top1_agreement': (student.argmax(-1) == teacher.argmax(-1)).float().mean().item(),
                    'mapping_cache_and_suffix_ms': (time.perf_counter() - start) * 1000}
            rows.append(row)
    summary = {method: {metric: sum(r['methods'][method][metric] for r in rows) / len(rows)
                         for metric in ('kl', 'top1_agreement', 'mapping_cache_and_suffix_ms')}
               for method in ('affine', 'attention')}
    return {'summary': summary, 'samples': rows,
            'timing_scope': 'single GPU; CPU source features to GPU, mapping, RoPE/cache creation and teacher-forced suffix included; source prefill and file I/O excluded; exploratory single measurements, not speedup evidence'}


def upload_outputs(args):
    api = HfApi()
    if not api.dataset_info(REPO).private:
        raise ValueError('Refusing public checkpoint storage')
    names = [p.name for p in (args.work / 'outputs').iterdir() if p.is_file()]
    commit = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=args.work / 'outputs',
            path_in_repo=args.upload_prefix, allow_patterns=names, commit_message=f'Cache attention: {args.work.name}')
    info = api.dataset_info(REPO, revision=commit.oid, files_metadata=True)
    if not info.private:
        raise ValueError('Checkpoint destination is no longer private')
    remote = {s.rfilename: s for s in info.siblings}
    for name in names:
        path = args.work / 'outputs' / name
        entry = remote[f'{args.upload_prefix}/{name}']
        if entry.lfs:
            matches = entry.lfs.sha256 == base.file_sha(path)
        else:
            import hashlib
            content = path.read_bytes()
            matches = entry.blob_id == hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if not matches:
            raise ValueError('Checkpoint upload verification failed; local outputs retained')
    base.write_json(args.work / 'upload-receipt.json', {'revision': commit.oid, 'prefix': args.upload_prefix})
    print(f'Outputs verified on Hugging Face: {commit.oid}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--source', choices=base.MODELS, default='smol')
    p.add_argument('--target', choices=base.MODELS, default='qwen')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--local-data', type=Path, help='Existing local pilot directory; no remote feature download')
    p.add_argument('--data-root', type=Path, help='Shared server feature cache for pilot and full runs')
    p.add_argument('--pilot', action='store_true', help='Four train / two validation prompts for runtime measurement')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=16, help='Aligned text positions per batch, not prompts')
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ridge', type=float, default=1e-2)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--kl-steps', type=int, default=100)
    p.add_argument('--probes', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--hours', type=float, default=8, help='Soft training budget, checked between epochs/steps')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--upload-prefix', help='Unique experiments/... destination; uploads after each epoch and at completion')
    args = p.parse_args()
    if args.data_root is None:
        args.data_root = args.work / 'data'
    if min(args.epochs, args.batch, args.stride, args.width, args.attention_heads, args.probes, args.patience) < 1 or args.kl_steps < 0 or args.hours <= 0 or args.ridge <= 0 or args.lr <= 0:
        p.error('Invalid training budget or hyperparameters')
    if args.source == args.target or args.width % args.attention_heads:
        p.error('Use different models and a width divisible by attention heads')
    if args.device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        p.error('Set CUDA_VISIBLE_DEVICES to one allocated GPU')
    if args.upload_prefix and (not args.upload_prefix.startswith('experiments/') or '..' in args.upload_prefix):
        p.error('Use a unique experiments/... upload prefix')
    import fcntl
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / 'training.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = args.work / 'outputs'
    out.mkdir(exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        p.error('Existing checkpoint: use --resume or a new work directory')
    if args.upload_prefix and not args.resume:
        if any(s.rfilename.startswith(args.upload_prefix + '/') for s in HfApi().dataset_info(REPO).siblings):
            p.error('Upload prefix already exists; use --resume with its work directory or a new prefix')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    data = Dataset(args)
    sg, tg = base.EXPECTED_GEOMETRY[args.source], base.EXPECTED_GEOMETRY[args.target]
    if sg['head_dim'] != tg['head_dim']:
        raise ValueError('This experiment requires equal head dimensions')
    config = {'source_layers': sg['layers'], 'source_heads': sg['kv_heads'], 'target_layers': tg['layers'],
              'target_heads': tg['kv_heads'], 'head_dim': sg['head_dim'], 'width': args.width,
              'attention_heads': args.attention_heads, 'source_id': list(base.MODELS).index(args.source),
              'target_id': list(base.MODELS).index(args.target)}
    run = {'config': config, 'source': args.source, 'target': args.target, 'models': base.MODELS,
           'manifest_sha256': data.manifest_sha, 'dataset_revision': REVISION if data.remote else 'local',
           'train_ids': [s['id'] for s in data.train], 'validation_ids': [s['id'] for s in data.validation],
           'batch': args.batch, 'stride': args.stride, 'ridge': args.ridge, 'seed': args.seed,
           'lr': args.lr, 'probes': args.probes, 'torch': str(torch.__version__),
           'epochs_requested': args.epochs, 'kl_steps_requested': args.kl_steps, 'patience': args.patience,
           'training_script_sha256': base.file_sha(Path(__file__)),
           'architecture_script_sha256': base.file_sha(Path(__file__).with_name('cache_attention.py')),
           'objective': 'train-only standardized KV MSE then frozen-receiver suffix KL + 0.01 KV anchoring',
           'conditioning_scope': 'one trained direction; model IDs do not demonstrate universal translation'}
    mapper = CacheAttention(config).to(args.device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch, best, stale, stage = 0, math.inf, 0, 'mse'
    if args.resume:
        saved = torch.load(out / 'last.pt', map_location=args.device, weights_only=True)
        if saved['run'] != run:
            raise ValueError('Checkpoint belongs to a different dataset or training configuration')
        mapper.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, best, stale, stage = saved['epoch'], saved['best'], saved['stale'], saved['stage']
    else:
        positions = fit_statistics_and_ridge(mapper, data, args)
        base.write_json(out / 'run.json', dict(run, parameters=sum(p.numel() for p in mapper.parameters()),
                        training_positions=positions, preparation_seconds=time.perf_counter() - started))
        checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run})
    def save(stage, ep, best, stale):
        checkpoint(out / 'last.pt', {'model': mapper.state_dict(), 'optimizer': optimizer.state_dict(),
                   'run': run, 'epoch': ep, 'best': best, 'stale': stale, 'stage': stage})
        if args.upload_prefix:
            upload_outputs(args)
    epoch_seconds = []
    resuming_kl = args.resume and (out / 'kl-last.pt').exists()
    if stage == 'mse' and not resuming_kl:
        for ep in range(epoch, args.epochs):
            if time.perf_counter() - started > args.hours * 3600 or stale >= args.patience:
                break
            epoch_start = time.perf_counter()
            mapper.train()
            order = list(data.train)
            random.Random(args.seed + ep).shuffle(order)
            losses, steps = 0.0, 0
            for sample in order:
                x, y = data.pair(sample, args.stride)
                permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed + ep + int(sample['id'][:6], 16)))
                for indices in permutation.split(args.batch):
                    xb, yb = x[indices].to(args.device), y[indices].to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(mapper(xb), mapper.standardize_target(yb))
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses += loss.item()
                    steps += 1
            metrics = validation_mse(mapper, data, args)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError('Non-finite validation loss')
            elapsed = time.perf_counter() - epoch_start
            epoch_seconds.append(elapsed)
            row = {'stage': 'mse', 'epoch': ep + 1, 'train_mse': losses / steps, 'validation': metrics,
                   'epoch_seconds': elapsed, 'remaining_epoch_estimate_seconds': sum(epoch_seconds) / len(epoch_seconds) * (args.epochs - ep - 1)}
            print(json.dumps(row), flush=True)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            if metrics['attention'] < best:
                best, stale = metrics['attention'], 0
                checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run})
            else:
                stale += 1
            save('mse', ep + 1, best, stale)
    mapper.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True)['model'])
    gc.collect()
    spec = base.MODELS[args.target]
    receiver = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
    receiver.requires_grad_(False)
    kl_optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr / 3, weight_decay=1e-4)
    kl_start = 0
    if resuming_kl:
        saved = torch.load(out / 'kl-last.pt', map_location=args.device, weights_only=True)
        if saved['run'] != run:
            raise ValueError('KL checkpoint configuration mismatch')
        mapper.load_state_dict(saved['model'])
        kl_optimizer.load_state_dict(saved['optimizer'])
        kl_start, best_kl = saved['step'], saved['best_kl']
    else:
        behavior = behavior_evaluation(mapper, receiver, data, args)
        base.write_json(out / 'behavior-mse.json', behavior)
        best_kl = behavior['summary']['attention']['kl']
        checkpoint(out / 'behavior-best.pt', {'model': mapper.state_dict(), 'run': run, 'validation_kl': best_kl})
    for step in range(kl_start, args.kl_steps):
        if time.perf_counter() - started > args.hours * 3600:
            break
        mapper.train()
        sample = data.train[step % len(data.train)]
        source = data.features(sample, args.source)
        target = data.features(sample, args.target)
        ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
        length = ids.shape[1] - args.probes
        with torch.no_grad():
            teacher = receiver(input_ids=ids, use_cache=False, logits_to_keep=args.probes).logits
        kl_optimizer.zero_grad(set_to_none=True)
        mapped = translate_prefix(mapper, source, sample, args, length)
        student = receiver_logits(receiver, mapped, ids[:, -args.probes:])
        kl = logit_kl(student, teacher)
        anchor = F.mse_loss(mapper.standardize_target(mapped), mapper.standardize_target(target[:length].to(args.device)))
        loss = kl + .01 * anchor
        if not torch.isfinite(loss):
            raise ValueError('Non-finite behavior calibration loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0, error_if_nonfinite=True)
        kl_optimizer.step()
        print(f'KL step {step+1}/{args.kl_steps}: {kl.item():.6f}', flush=True)
        if (step + 1) % 25 == 0 or step + 1 == args.kl_steps:
            behavior = behavior_evaluation(mapper, receiver, data, args)
            base.write_json(out / 'behavior-latest.json', behavior)
            if behavior['summary']['attention']['kl'] < best_kl:
                best_kl = behavior['summary']['attention']['kl']
                checkpoint(out / 'behavior-best.pt', {'model': mapper.state_dict(), 'run': run, 'validation_kl': best_kl})
        checkpoint(out / 'kl-last.pt', {'model': mapper.state_dict(), 'optimizer': kl_optimizer.state_dict(),
                   'run': run, 'step': step + 1, 'best_kl': best_kl})
        if ((step + 1) % 25 == 0 or step + 1 == args.kl_steps) and args.upload_prefix:
            upload_outputs(args)
    mapper.load_state_dict(torch.load(out / 'behavior-best.pt', map_location=args.device, weights_only=True)['model'])
    final = behavior_evaluation(mapper, receiver, data, args)
    final['wall_seconds'] = time.perf_counter() - started
    final['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
    base.write_json(out / 'final.json', final)
    if args.upload_prefix:
        upload_outputs(args)
    print('COMPLETE: ' + json.dumps(final['summary']), flush=True)


if __name__ == '__main__':
    main()
