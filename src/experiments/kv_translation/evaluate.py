# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0"]
# ///
"""Replay the held-out SmolLM2-to-Qwen context-only handoff."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mapper import capture, split_prompt, translate, to_cache
from semantic import score_answers

REPO = 'tandemkv/paired-kv-qwen-smollm'
REVISION = '27744a02d7d5f0abe61216612c46ec3b2a0bb52e'
RUN = 'experiments/ridge-32k.C70bPO'
PARENT_REVISION = 'ba8bfb17f27e5e6e10571093814c75a0c1fbef5c'
MODELS = {
    'smol': ('HuggingFaceTB/SmolLM2-360M-Instruct', 'a10cc1512eabd3dde888204e902eca88bddb4951'),
    'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775'),
}
CHECKPOINTS = {
    'ridge': ('ridge-8000.pt', '84046a87b50ddfb3423ceece930ed7ca2b7055cda09d48b8435c91b74eb54193'),
    'distilled': ('distilled-8000.pt', 'db99621579ef552721ffdf57b5308314d32e4944f87f2038f640ff65909bb0d6'),
}


def download(path, revision=REVISION, sha256=None):
    file = Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision))
    if sha256:
        with file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                raise ValueError(f'Checksum mismatch: {path}')
    return file


def forward(model, ids, cache=None):
    tokens = torch.tensor([ids], device=model.device)
    past = cache.get_seq_length() if cache is not None else 0
    positions = torch.arange(past, past + len(ids), device=model.device)
    return model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 cache_position=positions, position_ids=positions[None],
                 attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long, device=model.device))


def answer(model, tokenizer, start):
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    tokens = []
    for i in range(32):
        token = start.logits[0, -1].argmax().item()
        tokens.append(token)
        if token in eos or i == 31:
            break
        start = forward(model, [token], start.past_key_values)
    return {'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'tokens': tokens}


@torch.inference_mode()
def evaluate(count, device, output):
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError(output)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    path = download(f'{RUN}/test.jsonl.gz')
    manifest = json.loads(download(f'{RUN}/data.json').read_text())
    with gzip.open(path, 'rb') as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != manifest['sha256']['test.jsonl']:
        raise ValueError('Test split did not match its manifest')
    rows = [json.loads(line) for line in data.splitlines()]
    rows = [row for row in rows if row['dataset'] == 'hotpot']
    random.Random(20261113).shuffle(rows)
    if not 1 <= count <= len(rows):
        raise ValueError(f'Count must be between 1 and {len(rows)}')
    parent = download(
        'experiments/continuation-ada-full-tokens-32-20261029/best.pt', PARENT_REVISION,
        '713c5c140c01d42b26514025c4624a74d4493fa175243e4eb202a30c852371ad',
    )
    normalization = torch.load(parent, map_location=device, weights_only=True)['model']
    states = {
        name: torch.load(download(f'{RUN}/outputs/{file}', sha256=digest),
                         map_location=device, weights_only=True)
        for name, (file, digest) in CHECKPOINTS.items()
    }
    tokenizers = {k: AutoTokenizer.from_pretrained(model, revision=revision)
                  for k, (model, revision) in MODELS.items()}
    models = {k: AutoModelForCausalLM.from_pretrained(
        model, revision=revision, dtype=torch.float32, attn_implementation='sdpa').to(device).eval()
        for k, (model, revision) in MODELS.items()}
    qwen, smol = models['qwen'], models['smol']
    results = []
    for i, row in enumerate(rows[:count], 1):
        source_ids, target_ids, suffix, alignment = split_prompt(row['prompt'], tokenizers)
        native = forward(qwen, target_ids + suffix)
        native_features = capture(qwen, target_ids)
        replay = forward(qwen, suffix, to_cache(qwen, native_features))
        torch.testing.assert_close(replay.logits, native.logits, rtol=1e-3, atol=1e-3)
        source = capture(smol, source_ids)
        result = {'id': row['id'], 'question': row['question'], 'reference': row['answer']}
        result['native'] = answer(qwen, tokenizers['qwen'], native)
        result['question_only'] = answer(qwen, tokenizers['qwen'], forward(qwen, suffix))
        for name, state in states.items():
            features = translate(source, alignment, state, normalization, smol.config.num_key_value_heads)
            result[name] = answer(qwen, tokenizers['qwen'], forward(qwen, suffix, to_cache(qwen, features)))
        results.append(result)
        print(f'{i}/{count} {row["id"]}', flush=True)
    with output.open('x') as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('--count', type=int, default=120)
    replay.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    replay.add_argument('--output', type=Path, required=True)
    score = commands.add_parser('score')
    score.add_argument('answers', type=Path)
    score.add_argument('--method', choices=('distilled', 'ridge', 'question_only'), default='distilled')
    args = parser.parse_args()
    if args.command == 'replay':
        evaluate(args.count, args.device, args.output)
    else:
        content = args.answers.read_text()
        rows = json.loads(content) if content.lstrip().startswith('[') else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
        print(json.dumps(score_answers(rows, args.method), indent=2))
