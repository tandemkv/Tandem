# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Free-running and warm-timing evaluation of the existing mapper checkpoint."""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import paired_kv_dataset as base
from cache_attention import CacheAttention
from train_cache_attention import Dataset, REPO, logit_kl, synchronize, to_cache, translate_prefix

CHECKPOINT_REVISION = '3718426b6bd55c354d49912d342958305ac5fe10'
CHECKPOINT = 'experiments/attention-20261028-221854/full/behavior-best.pt'


def native_start(receiver, tokens):
    return receiver(input_ids=tokens, use_cache=True, logits_to_keep=1)


def cached_start(receiver, features, tokens):
    cache = to_cache(features, receiver)
    position = torch.arange(len(features), len(features) + tokens.shape[1], device=tokens.device)
    return receiver(input_ids=tokens, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, len(features) + tokens.shape[1], dtype=torch.long, device=tokens.device),
                    position_ids=position[None], cache_position=position, logits_to_keep=1)


def handoff_prefix_length(sample, target, handoff):
    length = len(sample['tokens'][target]['ids'])
    start = length - 1 if handoff == 'last-token' else min(sample['question_positions'][target])
    if not 0 < start < length:
        raise ValueError('Handoff requires a nonempty cache prefix and prompt suffix')
    return start


def replace_cache_component(mapped, native, component):
    """Oracle intervention in pre-RoPE feature space; never modify either input."""
    if mapped.shape != native.shape or mapped.ndim != 4 or mapped.shape[1] != 2:
        raise ValueError('Expected matching [tokens, K/V, layer-heads, head-dim] features')
    if component not in ('keys', 'values'):
        raise ValueError('Choose native keys or native values')
    return torch.stack((native[:, 0] if component == 'keys' else mapped[:, 0],
                        native[:, 1] if component == 'values' else mapped[:, 1]), dim=1)


def continue_greedy(receiver, output, max_new_tokens, eos_ids):
    cache, logits = output.past_key_values, output.logits[:, -1, :]
    tokens = []
    for i in range(max_new_tokens):
        token = logits.argmax(-1).item()
        tokens.append(token)
        if token in eos_ids or i == max_new_tokens - 1:
            break
        position = torch.tensor([cache.get_seq_length()], device=logits.device)
        output = receiver(input_ids=torch.tensor([[token]], device=logits.device), past_key_values=cache,
                          attention_mask=torch.ones(1, position.item() + 1, dtype=torch.long, device=logits.device),
                          position_ids=position[None], cache_position=position, use_cache=True, logits_to_keep=1)
        cache, logits = output.past_key_values, output.logits[:, -1, :]
    return tokens


def prefix_agreement(native, candidate):
    same = 0
    for a, b in zip(native, candidate):
        if a != b:
            break
        same += 1
    return {'first_token_match': bool(native and candidate and native[0] == candidate[0]),
            'matching_prefix_tokens': same, 'sequence_equal': native == candidate}


def timed(fn, device, repeats):
    fn()  # Separate warm-up for each path.
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return times


def lock_workdir(work):
    import fcntl
    lock = (work / 'evaluation.lock').open('w')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError(f'Another evaluation is using {work}') from None
    return lock


def check_destination(api, prefix):
    current = api.dataset_info(REPO, files_metadata=True)
    if not current.private:
        raise ValueError('Evaluation destination is not private')
    if any(s.rfilename.startswith(prefix + '/') for s in current.siblings):
        raise ValueError('Evaluation upload prefix already exists')
    return current


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--limit', type=int, default=40)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--handoff', choices=('last-token', 'question'), default='last-token',
                   help='Replay the final prompt token, or the question through Answer:, over the translated prefix')
    p.add_argument('--kv-ablation', action='store_true',
                   help='Also replace translated attention keys or values with native target features; diagnostic only')
    p.add_argument('--compare-ridge-layers', action='store_true',
                   help='Fit and evaluate a three-layer ridge map alongside the saved one-layer map and attention')
    p.add_argument('--calibrated-ridge-file', type=Path,
                   help='Compare a saved three-layer ridge map before and after calibration')
    p.add_argument('--checkpoint-file', type=Path, help='Evaluate a saved local mapper instead of the original cloud checkpoint')
    p.add_argument('--upload-prefix', help='Unique experiments/... destination')
    args = p.parse_args()
    if args.compare_ridge_layers and args.calibrated_ridge_file:
        p.error('Use either --compare-ridge-layers or --calibrated-ridge-file')
    if min(args.limit, args.tokens, args.repeats) < 1 or args.limit > 40:
        p.error('Invalid evaluation limits')
    if args.device == 'cuda' and (not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        p.error('Set CUDA_VISIBLE_DEVICES to one allocated GPU')
    if args.upload_prefix and (not args.upload_prefix.startswith('experiments/') or '..' in args.upload_prefix):
        p.error('Use an experiments/... upload prefix')
    args.work.mkdir(parents=True, exist_ok=True)
    lock = lock_workdir(args.work)
    output_path = args.work / 'evaluation.json'
    if output_path.exists():
        p.error('Output exists; use a new work directory')
    torch.set_num_threads(4)
    api = HfApi()
    if args.upload_prefix:
        check_destination(api, args.upload_prefix)
    if args.checkpoint_file:
        path = args.checkpoint_file
        checkpoint_sha = base.file_sha(path)
        checkpoint_revision, checkpoint_name = None, str(path.resolve())
    else:
        info = api.dataset_info(REPO, revision=CHECKPOINT_REVISION, files_metadata=True)
        if not info.private:
            raise ValueError('Expected private checkpoint repository')
        entry = next(s for s in info.siblings if s.rfilename == CHECKPOINT)
        path = Path(hf_hub_download(REPO, CHECKPOINT, repo_type='dataset', revision=CHECKPOINT_REVISION))
        if not entry.lfs or base.file_sha(path) != entry.lfs.sha256:
            raise ValueError('Checkpoint hash mismatch')
        checkpoint_sha = entry.lfs.sha256
        checkpoint_revision, checkpoint_name = CHECKPOINT_REVISION, CHECKPOINT
    saved = torch.load(path, map_location=args.device, weights_only=True)
    run = saved['run']
    if (run['source'], run['target'], run['training_script_sha256'], run['architecture_script_sha256']) != (
            'smol', 'qwen', base.file_sha(Path(__file__).with_name('train_cache_attention.py')),
            base.file_sha(Path(__file__).with_name('cache_attention.py'))):
        raise ValueError('Checkpoint code or model identity mismatch')
    dataset_args = SimpleNamespace(local_data=None, data_root=args.data_root, work=args.work,
                                   source=run['source'], target=run['target'], pilot=False)
    data = Dataset(dataset_args)
    if data.manifest_sha != run['manifest_sha256'] or [s['id'] for s in data.validation] != run['validation_ids']:
        raise ValueError('Validation dataset does not match checkpoint')
    mapper = CacheAttention(run['config']).to(args.device)
    mapper.load_state_dict(saved['model'])
    mapper.eval()
    ridge_metadata = None
    if args.compare_ridge_layers:
        from cache_ridge import ThreeLayerRidge

        if [s['id'] for s in data.train] != run['train_ids']:
            raise ValueError('Training split does not match the saved ridge fit')
        print(f'Fitting three-layer ridge on {len(data.train)} training prompts', flush=True)
        started = time.perf_counter()
        three_layer = ThreeLayerRidge(mapper)
        positions = three_layer.fit(data, run['stride'], run['ridge'])
        ridge_metadata = {'training_positions': positions, 'training_prompts': len(data.train),
                          'stride': run['stride'], 'penalty': run['ridge'],
                          'source_layers_by_target': three_layer.layers.tolist(),
                          'coefficients': three_layer.weights.numel(),
                          'fit_seconds': time.perf_counter() - started,
                          'script_sha256': base.file_sha(Path(__file__).with_name('cache_ridge.py'))}
        print(f'Ridge fit complete: {ridge_metadata["fit_seconds"]:.1f}s', flush=True)
    calibration_metadata = None
    if args.calibrated_ridge_file:
        from cache_ridge import ThreeLayerRidge

        folded = torch.load(args.calibrated_ridge_file, map_location=args.device, weights_only=True)
        if folded['parent_sha256'] != checkpoint_sha or folded['manifest_sha256'] != data.manifest_sha:
            raise ValueError('Calibrated ridge map does not match the checkpoint or dataset')
        three_layer = ThreeLayerRidge(mapper)
        calibrated = ThreeLayerRidge(mapper)
        shape = (2, len(three_layer.layers), 3 * three_layer.source_width + 1, three_layer.target_width)
        if not torch.equal(three_layer.layers, folded['source_layers']):
            raise ValueError('Calibrated ridge layer selection differs')
        for translator, key in ((three_layer, 'ridge'), (calibrated, 'calibrated')):
            weights = folded[key]
            if weights.shape != shape or weights.dtype != torch.float32 or not torch.isfinite(weights).all():
                raise ValueError(f'Invalid {key} weights')
            translator.weights = weights
        calibration_metadata = {key: folded[key] for key in ('rank', 'best_epoch', 'train_prompts',
                                'validation_prompts', 'pilot', 'trainable_parameters', 'seconds')}
        calibration_metadata['map_sha256'] = base.file_sha(args.calibrated_ridge_file)
    spec = base.MODELS[run['target']]
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
    receiver = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], dtype=torch.float32,
                                                    attn_implementation='eager').eval().to(args.device)
    receiver.requires_grad_(False)
    eos = receiver.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    translation_args = SimpleNamespace(source=run['source'], target=run['target'], device=args.device, batch=run['batch'])
    methods = {'affine': (mapper, True, None), 'attention': (mapper, False, None)}
    if args.compare_ridge_layers or args.calibrated_ridge_file:
        methods['three_layer_ridge'] = (three_layer, True, None)
    if args.calibrated_ridge_file:
        methods['calibrated_ridge'] = (calibrated, True, None)
    if args.kv_ablation:
        methods.update(native_keys_mapped_values=(mapper, False, 'keys'),
                       mapped_keys_native_values=(mapper, False, 'values'))
    rows = []
    for sample in data.validation[:args.limit]:
        source = data.features(sample, run['source'])
        target = data.features(sample, run['target'])
        tokens = torch.tensor([sample['tokens'][run['target']]['ids']], device=args.device)
        prefix_len = handoff_prefix_length(sample, run['target'], args.handoff)
        suffix = tokens[:, prefix_len:]
        with torch.inference_mode():
            native = native_start(receiver, tokens)
            control = cached_start(receiver, target[:prefix_len].to(args.device), suffix)
            control_error = (native.logits - control.logits).abs().max().item()
            if not torch.allclose(native.logits, control.logits, rtol=1e-3, atol=1e-3):
                raise ValueError(f'Native cache replay failed: {sample["id"]}')
            native_tokens = continue_greedy(receiver, native, args.tokens, eos_ids)
            control_tokens = continue_greedy(receiver, control, args.tokens, eos_ids)
            if native_tokens != control_tokens:
                raise ValueError(f'Native cache continuation diverged: {sample["id"]}')
            row = {'id': sample['id'], 'prompt_tokens': tokens.shape[1], 'control_max_logit_error': control_error,
                   'replayed_prompt_tokens': suffix.shape[1],
                   'native_tokens': native_tokens, 'native_text': tokenizer.decode(native_tokens, skip_special_tokens=True),
                   'methods': {}}
            def native_fn():
                return native_start(receiver, tokens)
            row['native_ms'] = timed(native_fn, args.device, args.repeats)
            for name, (translator, affine, native_component) in methods.items():
                def mapped_fn():
                    features = translate_prefix(translator, source, sample, translation_args, prefix_len, affine)
                    if native_component:
                        features = replace_cache_component(features, target[:prefix_len].to(args.device), native_component)
                    return cached_start(receiver, features, suffix)
                measurements = [] if native_component else timed(mapped_fn, args.device, args.repeats)
                output = mapped_fn()
                first_kl = logit_kl(output.logits, native.logits).item()
                if not math.isfinite(first_kl):
                    raise ValueError(f'Nonfinite first-token KL: {sample["id"]}, {name}')
                native_token = native.logits[0, -1].argmax()
                native_rank = 1 + (output.logits[0, -1] > output.logits[0, -1, native_token]).sum().item()
                generated = continue_greedy(receiver, output, args.tokens, eos_ids)
                row['methods'][name] = dict(prefix_agreement(native_tokens, generated), tokens=generated,
                    oracle_native_component=native_component,
                    first_token_kl=first_kl, native_first_token_rank=native_rank,
                    text=tokenizer.decode(generated, skip_special_tokens=True), handoff_ms=measurements)
            rows.append(row)
        print(f'Evaluated {len(rows)}/{min(args.limit, len(data.validation))}: {sample["id"]}', flush=True)
    report = {'dataset_revision': run['dataset_revision'], 'checkpoint_revision': checkpoint_revision,
              'checkpoint': checkpoint_name, 'checkpoint_sha256': checkpoint_sha, 'split': 'validation',
              'handoff': args.handoff,
              'three_layer_ridge_fit': ridge_metadata,
              'ridge_calibration': calibration_metadata,
              'oracle_kv_ablation': args.kv_ablation,
              'oracle_note': 'Native-component substitutions require receiver prefill and are diagnostic only. They test interacting components, not an additive decomposition of error. Oracle methods are not timed.' if args.kv_ablation else None,
              'note': 'Development prompts used for checkpoint selection; not a held-out final test.',
              'generation': {'mode': 'greedy', 'max_new_tokens': args.tokens, 'eos_ids': sorted(eos_ids)},
              'timing': {'repeats': args.repeats, 'warmups': 1,
                  'scope': 'native GPU prefill to first-token logits vs CPU source-cache mapping, alignment, CPU-to-GPU copy, RoPE/cache construction and first-token logits on one GPU; excludes source prefill, disk I/O and tokenization; does not represent GPU-to-GPU handoff'},
              'samples': rows}
    report['summary'] = {'prompts': len(rows), 'native_median_ms': statistics.median(statistics.median(r['native_ms']) for r in rows),
        'native_immediate_eos': sum(r['native_tokens'][0] in eos_ids for r in rows), 'methods': {}}
    for name in methods:
        report['summary']['methods'][name] = {'first_token_agreement': sum(r['methods'][name]['first_token_match'] for r in rows) / len(rows),
            'mean_first_token_kl': statistics.mean(r['methods'][name]['first_token_kl'] for r in rows),
            'exact_sequence_agreement': sum(r['methods'][name]['sequence_equal'] for r in rows) / len(rows),
            'mean_matching_prefix_tokens': statistics.mean(r['methods'][name]['matching_prefix_tokens'] for r in rows),
            'median_handoff_ms': None if methods[name][2] else statistics.median(statistics.median(r['methods'][name]['handoff_ms']) for r in rows),
            'immediate_eos': sum(r['methods'][name]['tokens'][0] in eos_ids for r in rows)}
    base.write_json(output_path, report)
    if args.upload_prefix:
        current = check_destination(api, args.upload_prefix)
        commit = api.upload_file(repo_id=REPO, repo_type='dataset', path_or_fileobj=str(output_path),
                                 path_in_repo=f'{args.upload_prefix}/evaluation.json', parent_commit=current.sha,
                                 commit_message='Add free-running cache mapper evaluation')
        info = api.dataset_info(REPO, revision=commit.oid, files_metadata=True)
        entry = next(s for s in info.siblings if s.rfilename == f'{args.upload_prefix}/evaluation.json')
        import hashlib
        content = output_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest() if entry.lfs else hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if not info.private or digest != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError('Evaluation upload verification failed; local report retained')
        print(f'Uploaded and verified evaluation: {commit.oid}', flush=True)
    print('COMPLETE: ' + json.dumps(report['summary']), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
