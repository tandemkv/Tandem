# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Compare saved calibration checkpoints at 32 tokens and publish existing outputs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import paired_kv_dataset as base
from evaluate_cache_attention import lock_workdir
from calibrate_cache_attention import CheckpointPublisher


def check_report(report, checkpoint):
    if (report['checkpoint_sha256'] != base.file_sha(checkpoint) or
            report.get('handoff', 'last-token') != 'last-token' or
            report['generation']['mode'] != 'greedy' or
            report['generation']['max_new_tokens'] != 32 or
            report['split'] != 'validation' or len(report['samples']) != 40 or
            len({s['id'] for s in report['samples']}) != 40):
        raise ValueError('Existing evaluation does not match this checkpoint and 40-prompt, 32-token comparison')


def compare_reports(reports):
    left, right = reports[1], reports[32]
    if left['dataset_revision'] != right['dataset_revision']:
        raise ValueError('Evaluation datasets differ')
    native = lambda report: {s['id']: s['native_tokens'] for s in report['samples']}
    if native(left) != native(right):
        raise ValueError('Native prompts or continuations differ between evaluations')
    return {str(tokens): report['summary']['methods']['attention'] for tokens, report in reports.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-home', type=Path, default=Path.home() / 'tandem-cluster/continuation-ada')
    args = p.parse_args()
    root = args.run_home
    root.mkdir(parents=True, exist_ok=True)
    lock = lock_workdir(root)
    data = Path((root / 'scratch-path.txt').read_text().strip()) / 'data'
    if not data.is_dir():
        raise ValueError('Existing Ada feature cache is missing')
    reports, uploads = {}, []
    for tokens in (1, 32):
        training = root / f'full-tokens-{tokens}'
        weights = training / 'outputs/best.pt'
        if not weights.is_file():
            raise ValueError(f'Missing selected checkpoint: {weights}')
        evaluation = root / f'evaluation-32-tokens-{tokens}'
        output = evaluation / 'outputs'
        report_path = evaluation / 'evaluation.json'
        if not report_path.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_cache_attention.py')),
                '--work', str(evaluation), '--data-root', str(data), '--checkpoint-file', str(weights),
                '--tokens', '32', '--limit', '40', '--repeats', '1'], check=True)
        report = json.loads(report_path.read_text())
        check_report(report, weights)
        output.mkdir(exist_ok=True)
        base.write_json(output / 'evaluation.json', report)
        reports[tokens] = report
        uploads.extend([(training, f'experiments/continuation-ada-full-tokens-{tokens}-20261029'),
                        (evaluation, f'experiments/continuation-ada-evaluation-32-tokens-{tokens}-20261029')])
    summary = compare_reports(reports)
    base.write_json(root / 'comparison-32-tokens.json', summary)
    print('COMPARISON: '+json.dumps(summary), flush=True)
    blocked_until, pending = 0., []
    for work, prefix in uploads:
        publisher = CheckpointPublisher(SimpleNamespace(work=work, upload_prefix=prefix))
        publisher.blocked_until = max(publisher.blocked_until, blocked_until)
        if not publisher(final=True):
            pending.append(str(work / 'outputs'))
        blocked_until = max(blocked_until, publisher.blocked_until)
    print('FINISHED: '+json.dumps({'pending_uploads': pending,
        'note': 'Rerun this command after the upload cooldown; valid saved evaluations are reused.' if pending else 'All four output folders uploaded and verified.'}), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
