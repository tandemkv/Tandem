# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0"]
# ///
"""Run an experiment, judge its answers, or rescore archived answers with BEM.

  run.py run context_only --out DIR        -> DIR/context_only-<time>/{config.json,rows.jsonl,summary.json}
  run.py judge DIR/context_only-<time>      -> adds judge verdicts and semantic agreement
  run.py bem answers.json --method distilled
"""
import argparse
import importlib
import json
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
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True, text=True)
    return head + ('-dirty' if dirty.stdout.strip() else '')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


@torch.inference_mode()
def run(experiment, count, device, dtype, out):
    spec = importlib.import_module(f'experiments.{experiment}')
    cases = spec.cases()[:count]
    directory = out / f'{experiment}-{time.strftime("%Y%m%d-%H%M%S")}'
    directory.mkdir(parents=True)
    h = harness.Harness(device, getattr(torch, dtype))
    write_json(directory / 'config.json', {
        'experiment': experiment, 'cases': len(cases), 'conditions': spec.CONDITIONS,
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
    write_json(directory / 'summary.json', summarize(rows))
    print(directory)


def read_rows(directory):
    return [json.loads(line) for line in (directory / 'rows.jsonl').read_text().splitlines()]


def judge(directory, device):
    from judge import JUDGE, Judge, annotate
    rows = annotate(read_rows(directory), Judge(device))
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(directory / 'summary.json', summarize(rows))
    config = json.loads((directory / 'config.json').read_text())
    write_json(directory / 'config.json', config | {'judge': JUDGE, 'judge_commit': git_commit()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('run')
    command.add_argument('experiment')
    command.add_argument('--count', type=int, default=120)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    command.add_argument('--out', type=Path, required=True, help='Parent directory for the run, outside the repo')
    command = commands.add_parser('judge')
    command.add_argument('directory', type=Path)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command = commands.add_parser('bem')
    command.add_argument('answers', type=Path)
    command.add_argument('--method', choices=('distilled', 'ridge', 'question_only'), default='distilled')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.experiment, args.count, args.device, args.dtype, args.out)
    elif args.command == 'judge':
        judge(args.directory, args.device)
    else:
        from semantic import score_answers
        content = args.answers.read_text()
        rows = json.loads(content) if content.lstrip().startswith('[') else [
            json.loads(line) for line in content.splitlines() if line.strip()]
        print(json.dumps(score_answers(rows, args.method), indent=2))
