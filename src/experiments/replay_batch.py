# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Test saved cache replay across prompt lengths and both data splits."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import paired_kv_dataset as base
from huggingface_hub import HfApi, hf_hub_download

REPO = 'tandemkv/paired-kv-qwen-smollm'
PREFIX = 'runs/hotpot-200-20261028'
REVISION = 'dca861cd61b10859d5682565cfbba5606e5c5974'


def select_samples(manifest, per_split=5):
    selected = []
    for split in ('train', 'validation'):
        samples = sorted((s for s in manifest['samples'] if s['split'] == split),
                         key=lambda s: (max(len(s['tokens'][model]['ids']) for model in base.MODELS), s['id']))
        if per_split < 2 or per_split > len(samples):
            raise ValueError('Need at least 2 examples per split and enough samples')
        selected.extend(samples[round(i * (len(samples) - 1) / (per_split - 1))] for i in range(per_split))
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    p.add_argument('--per-split', type=int, default=5)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if not info.private:
        raise ValueError('Expected a private dataset')
    entries = {s.rfilename: s for s in info.siblings}
    def download(name):
        return Path(hf_hub_download(REPO, f'{PREFIX}/{name}', repo_type='dataset', revision=REVISION, local_dir=args.work))
    manifest_path = download('manifest.json')
    checkpoint_path = download('checkpoint.json')
    manifest = json.loads(manifest_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    base.validate_manifest(manifest)
    if base.file_sha(manifest_path) != checkpoint['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    samples = select_samples(manifest, args.per_split)
    data_dir = args.work / PREFIX
    report = {'revision': REVISION, 'selected': [{'id': s['id'], 'split': s['split'],
        'tokens': {m: len(s['tokens'][m]['ids']) for m in base.MODELS}} for s in samples], 'results': []}
    for model in base.MODELS:
        for sample in samples:
            name = f'{model}/{sample["id"]}.safetensors'
            record = checkpoint['shards'][name]
            entry = entries[f'{PREFIX}/{name}']
            if entry.lfs is None or entry.lfs.sha256 != record['sha256']:
                raise ValueError(f'Remote checksum mismatch: {name}')
            path = download(name)
            if base.file_sha(path) != record['sha256']:
                raise ValueError(f'Download checksum mismatch: {name}')
            print(f'Testing {model} {sample["id"]} ({sample["split"]}, {len(sample["tokens"][model]["ids"])} tokens)', flush=True)
            cmd = [sys.executable, str(Path(__file__).with_name('replay_cache.py')), '--dataset', str(data_dir), '--model', model, '--sample', sample['id'], '--device', args.device]
            result = subprocess.run(cmd, text=True, capture_output=True)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr, flush=True)
                raise SystemExit(f'FAIL: {model} {sample["id"]}; downloaded shard retained at {path}')
            report['results'].append({'model': model, 'sample': sample['id'], 'output': result.stdout.strip()})
            path.unlink()
            (args.work / 'replay-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(report["results"])} replay checks across {len(samples)} prompts; report: {args.work / "replay-report.json"}', flush=True)


if __name__ == '__main__':
    main()
