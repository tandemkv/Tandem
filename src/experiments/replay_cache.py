# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Compare saved pre-RoPE KV replay against native prefill for one prompt."""
import argparse
import json
from pathlib import Path

import paired_kv_dataset as base
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, DynamicCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Directory with manifest.json and model shard folders')
    p.add_argument('--model', choices=base.MODELS, required=True)
    p.add_argument('--sample', help='Sample ID; defaults to first manifest sample')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    base.validate_manifest(manifest)
    sample = next(s for s in manifest['samples'] if not args.sample or s['id'] == args.sample)
    ids = sample['tokens'][args.model]['ids']
    path = args.dataset / args.model / (sample['id'] + '.safetensors')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        identity = json.loads(f.metadata()['identity'])
    if identity['text_sha256'] != sample['text_sha256'] or identity['model'] != base.MODELS[args.model] or identity['feature_space'] != base.SPACE or identity['manifest_sha256'] != base.file_sha(manifest_path):
        raise ValueError('Shard identity does not match the manifest')
    tensors = load_file(str(path))
    shape = dict(base.EXPECTED_GEOMETRY[args.model], tokens=len(ids))
    base.validate_features(tensors, shape)
    spec = base.MODELS[args.model]
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
    if base.geometry(model.config) != base.EXPECTED_GEOMETRY[args.model]:
        raise ValueError('Unexpected model geometry')
    if args.model == 'qwen':
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    with torch.inference_mode():
        tokens = torch.tensor([ids], device=args.device)
        expected = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        native = model(input_ids=tokens[:, :-1], use_cache=True)
        native_logits = model(input_ids=tokens[:, -1:], past_key_values=native.past_key_values, use_cache=True).logits[:, -1, :]
        positions = torch.arange(len(ids)-1, device=args.device)[None, :]
        dummy = torch.zeros((1, 1, len(ids)-1, shape['head_dim']), device=args.device)
        cos, sin = model.model.rotary_emb(dummy, positions)
        cache = DynamicCache(config=model.config)
        for layer in range(shape['layers']):
            def reshape(role):
                return tensors[f'layer_{layer}.{role}'][:-1].to(args.device).reshape(1, -1, shape['kv_heads'], shape['head_dim']).transpose(1, 2).contiguous()
            key, value = reshape('k'), reshape('v')
            _, key = apply_rotary_pos_emb(key, key, cos, sin)
            cache.update(key, value, layer)
        replay = model(input_ids=tokens[:, -1:], past_key_values=cache, use_cache=True).logits[:, -1, :]
    for label, actual in [('native cache', native_logits), ('saved cache', replay)]:
        error = (expected - actual).abs().max().item()
        same_token = expected.argmax().item() == actual.argmax().item()
        print(f'{label}: max logit error={error:.6g}, same next token={same_token}', flush=True)
        if not torch.allclose(expected, actual, rtol=1e-3, atol=1e-3) or not same_token:
            raise SystemExit(f'FAIL: {label} does not match native prefill')
    print(f'PASS: {args.model} sample {sample["id"]}, {len(ids)} tokens')


if __name__ == '__main__':
    main()
