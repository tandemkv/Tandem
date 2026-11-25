# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "accelerate==1.9.0", "bitsandbytes==0.50.2",
#   "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0",
# ]
# ///
"""Run an experiment, judge its answers, or rescore archived answers with BEM.

  run.py run context_only --out DIR        -> DIR/context_only-<time>/{config.json,rows.jsonl,summary.json}
  run.py judge DIR/context_only-<time>      -> adds judge verdicts and semantic agreement
  run.py summarize DIR/context_only-<time>  -> recomputes summary.json, including subsets
  run.py merge OUT SHARD...                 -> one run from consecutive shards (run --start)
  run.py bem answers.json --method distilled
"""
import argparse
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import transformers

import harness
from conditions import CONDITIONS, Prepared
from metrics import compare, summarize

HERE = Path(__file__).resolve().parent


def git_commit():
    if 'TANDEM_COMMIT' in os.environ:  # Modal copies the code without git; modal_app.py passes the commit.
        return os.environ['TANDEM_COMMIT']
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True, text=True)
    return head + ('-dirty' if dirty.stdout.strip() else '')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


@torch.inference_mode()
def run(experiment, start, count, device, dtype, out):
    spec = importlib.import_module(f'experiments.{experiment}')
    cases = spec.cases()[start:start + count]
    directory = out / f'{experiment}-{time.strftime("%Y%m%d-%H%M%S")}'
    directory.mkdir(parents=True)
    h = harness.Harness(device, getattr(torch, dtype))
    write_json(directory / 'config.json', {
        'experiment': experiment, 'start': start, 'cases': len(cases), 'conditions': spec.CONDITIONS,
        'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu', 'dtype': dtype,
        'commit': git_commit(), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'dataset': [harness.REPO, harness.REVISION], 'models': harness.MODELS,
        'checkpoints': harness.CHECKPOINTS,
    })
    rows = []
    with (directory / 'rows.jsonl').open('x') as stream:
        for i, case in enumerate(cases, 1):
            p = Prepared(h, case)
            native, native_first, native_nll = h.check_replay(p.target_ids, p.suffix)
            row = {'id': case.id, 'question': case.question, 'gold': case.gold, 'native': native,
                   'conditions': {}}
            for name in spec.CONDITIONS:
                features = CONDITIONS[name](h, p)
                answer, first = h.generate(p.suffix, h.cache(features))
                nll = h.answer_nll(p.suffix, h.cache(features), native['tokens'])
                answer['metrics'] = compare(native, answer, native_first, first, native_nll, nll, case.gold)
                row['conditions'][name] = answer
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{i}/{len(cases)} {case.id}', flush=True)
    write_summary(directory, rows)
    print(directory)


def read_rows(directory):
    return [json.loads(line) for line in (directory / 'rows.jsonl').read_text().splitlines()]


def write_summary(directory, rows, **extra):
    """Summarize all rows and each subset the experiment declares, keeping earlier judge validation."""
    experiment = json.loads((directory / 'config.json').read_text())['experiment']
    spec = importlib.import_module(f'experiments.{experiment}')
    summary = summarize(rows)
    subsets = {name: [r for r in rows if keep(r)] for name, keep in getattr(spec, 'SUBSETS', {}).items()}
    if subsets:
        summary['subsets'] = {name: summarize(chosen) for name, chosen in subsets.items() if chosen}
    previous = directory / 'summary.json'
    if 'judge_validation' not in extra and previous.exists():
        extra = {k: v for k, v in json.loads(previous.read_text()).items() if k == 'judge_validation'}
    write_json(previous, summary | extra)


def merge(out, directories):
    """Join consecutive shards of one run, pooling judge validation by pair counts."""
    configs = [json.loads((d / 'config.json').read_text()) for d in directories]
    first = configs[0]
    for previous, config in zip(configs, configs[1:]):
        if config['start'] != previous['start'] + previous['cases']:
            raise ValueError('Shards must be consecutive')
    for key in ('experiment', 'commit', 'conditions', 'dtype', 'judge_commit'):
        if len({json.dumps(c.get(key)) for c in configs}) > 1:
            raise ValueError(f'Shards differ in {key}')
    rows = [row for d in directories for row in read_rows(d)]
    out.mkdir(parents=True)
    write_json(out / 'config.json', first | {'cases': len(rows), 'shards': [
        {'start': c['start'], 'cases': c['cases'], 'device': c['device']} for c in configs]})
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    validations = [json.loads((d / 'summary.json').read_text()).get('judge_validation') for d in directories]
    extra = {}
    if all(validations):
        pooled = {}
        for kind in ('same', 'different'):
            pairs = sum(v[f'{kind}_pairs'] for v in validations)
            pooled[f'{kind}_accuracy'] = sum(v[f'{kind}_accuracy'] * v[f'{kind}_pairs'] for v in validations) / pairs
            pooled[f'{kind}_pairs'] = pairs
        extra['judge_validation'] = pooled | {'misses': [m for v in validations for m in v['misses']]}
    elif any(validations):
        raise ValueError('Only some shards were judged')
    write_summary(out, rows, **extra)


def judge(directory, device):
    from judge import JUDGE, Judge, annotate, validate
    model, rows = Judge(device), read_rows(directory)
    validation = validate(rows, model)  # Accuracy on pairs whose verdict is known by construction.
    rows = annotate(rows, model)
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_summary(directory, rows, judge_validation=validation)
    print(json.dumps({k: v for k, v in validation.items() if k != 'misses'}, indent=2))
    config = json.loads((directory / 'config.json').read_text())
    write_json(directory / 'config.json', config | {'judge': JUDGE, 'judge_quantization': 'bitsandbytes nf4',
                                                    'judge_commit': git_commit()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('run')
    command.add_argument('experiment')
    command.add_argument('--start', type=int, default=0, help='First case, to run one shard')
    command.add_argument('--count', type=int, default=120)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    command.add_argument('--out', type=Path, required=True, help='Parent directory for the run, outside the repo')
    command = commands.add_parser('judge')
    command.add_argument('directory', type=Path)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command = commands.add_parser('summarize', help='Recompute summary.json, including subsets, from rows.jsonl')
    command.add_argument('directory', type=Path)
    command = commands.add_parser('merge', help='Join consecutive shard runs into one run')
    command.add_argument('out', type=Path)
    command.add_argument('shards', type=Path, nargs='+')
    command = commands.add_parser('bem')
    command.add_argument('answers', type=Path)
    command.add_argument('--method', choices=('distilled', 'ridge', 'question_only'), default='distilled')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.experiment, args.start, args.count, args.device, args.dtype, args.out)
    elif args.command == 'judge':
        judge(args.directory, args.device)
    elif args.command == 'merge':
        merge(args.out, args.shards)
    elif args.command == 'summarize':
        write_summary(args.directory, read_rows(args.directory))
    else:
        from semantic import score_answers
        content = args.answers.read_text()
        rows = json.loads(content) if content.lstrip().startswith('[') else [
            json.loads(line) for line in content.splitlines() if line.strip()]
        print(json.dumps(score_answers(rows, args.method), indent=2))
