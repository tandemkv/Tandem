# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Publish the exact scaling experiment prompts to the private Tandem dataset."""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

import paired_kv_dataset as base
from scale_ridge_distillation import rows
from train_cache_attention import REPO, REVISION


def recover(samples, source_split):
    pending = {s['id']: s for s in samples}
    result = {}
    for raw in rows(source_split, random.Random(0)):
        sample = pending.pop(raw['id'], None)
        if sample is None:
            continue
        prompt, _ = base.format_prompt(raw)
        if base.sha(prompt.encode()) != sample['text_sha256']:
            raise ValueError(f'Prompt hash differs for {sample["id"]}')
        result[sample['id']] = dict(sample, text=prompt)
        if not pending:
            return [result[s['id']] for s in samples]
    raise ValueError(f'Could not recover {len(pending)} prompts')


def pack(sample, in_small_arm, gold):
    return {
        'id': sample['id'], 'prompt': sample['text'], 'prompt_sha256': sample['text_sha256'],
        'document_hashes': sample['documents'], 'in_160_arm': in_small_arm, 'gold_answer': gold,
        'source_token_ids': sample['tokens']['smol']['ids'],
        'source_offsets': sample['tokens']['smol']['offsets'],
        'target_token_ids': sample['tokens']['qwen']['ids'],
        'target_offsets': sample['tokens']['qwen']['offsets'],
        'target_to_source': sample['alignment']['smol_to_qwen']['target_to_source'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--original-manifest', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    if split['pilot'] or base.file_sha(args.original_manifest) != split['original_manifest_sha256']:
        raise ValueError('Expected the full run and its original manifest')
    original = json.loads(args.original_manifest.read_text())
    base.validate_manifest(original)
    lookup = {s['id']: s for s in original['samples'] if s['split'] == 'train'}
    core = [lookup[id_] for id_ in split['original_train_ids']]
    extra = recover(split['added'], 'train')
    test = recover(split['test'], 'validation')
    train = core + extra
    test_docs = {d for s in test for d in s['documents']}
    all_training_docs = {d for s in original['samples'] + extra for d in s['documents']}
    if (len(core) != 160 or len(train) != 800 or len(test) != 120 or test_docs & all_training_docs
            or len({s['id'] for s in train + test}) != 920
            or len({s['text_sha256'] for s in train + test}) != 920):
        raise ValueError('Expected a nested 160/800 split and 120 document-disjoint test prompts')

    args.staging.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([
        ('id', pa.string()), ('prompt', pa.string()), ('prompt_sha256', pa.string()),
        ('document_hashes', pa.list_(pa.string())), ('in_160_arm', pa.bool_()), ('gold_answer', pa.string()),
        ('source_token_ids', pa.list_(pa.int32())), ('source_offsets', pa.list_(pa.list_(pa.int32()))),
        ('target_token_ids', pa.list_(pa.int32())), ('target_offsets', pa.list_(pa.list_(pa.int32()))),
        ('target_to_source', pa.list_(pa.int32())),
    ])
    pq.write_table(pa.Table.from_pylist([pack(s, i < len(core), None) for i, s in enumerate(train)], schema=schema),
                   args.staging / 'train.parquet')
    pq.write_table(pa.Table.from_pylist([pack(s, False, s['answer']) for s in test], schema=schema),
                   args.staging / 'test.parquet')
    files = [args.staging / filename for filename in ('train.parquet', 'test.parquet')]
    base.write_json(args.staging / 'manifest.json', {
        'source': base.SOURCE, 'models': base.MODELS, 'seed': split['seed'],
        'prompt_format': original['prompt_format'], 'original_manifest_revision': REVISION,
        'original_manifest_sha256': split['original_manifest_sha256'],
        'train_prompts': 800, 'small_arm_prompts': 160, 'test_prompts': 120,
        'training_updates_per_arm': split['steps_per_arm'], 'teacher_tokens': split['training_tokens'],
        'test_generation_tokens': split['evaluation_tokens'], 'ridge_sha256': split['ridge_sha256'],
        'sha256': {file.name: base.file_sha(file) for file in files},
        'license': base.SOURCE['license']})
    files.append(args.staging / 'manifest.json')

    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    if not info.private or any(s.rfilename.startswith(args.prefix + '/') for s in info.siblings):
        raise ValueError('Expected a fresh path in the private Tandem dataset')
    commit = api.create_commit(repo_id=REPO, repo_type='dataset', parent_commit=info.sha,
                               operations=[CommitOperationAdd(path_in_repo=f'{args.prefix}/{f.name}',
                                                              path_or_fileobj=f) for f in files],
                               commit_message='Add HotpotQA prompts and splits for ridge distillation')
    for file in files:
        downloaded = Path(hf_hub_download(REPO, f'{args.prefix}/{file.name}', repo_type='dataset',
                                          revision=commit.oid))
        if base.file_sha(downloaded) != base.file_sha(file):
            raise ValueError(f'Uploaded bytes differ for {file.name}')
    print(f'COMPLETE: https://huggingface.co/datasets/{REPO}/tree/{commit.oid}/{args.prefix}', flush=True)


if __name__ == '__main__':
    main()
