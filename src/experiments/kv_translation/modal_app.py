"""Run KV translation jobs on a Modal H100 and archive each output folder to the Tandem dataset.

Needs a Modal secret named `huggingface` with HF_TOKEN. The code must be committed: results record the commit.

  modal run --detach modal_app.py::train_keys --steps 4000   -> experiments/keyfit-<time>/
  modal run --detach modal_app.py::experiment --name key_fit --count 500
  modal run --detach modal_app.py::judge --run key_fit-<time>   (the folder the experiment call printed)
"""
import hashlib
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATASET = 'tandemkv/paired-kv-qwen-smollm'
RUNS = Path('/runs')

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.11.24')
    .add_local_file(HERE / 'run.py', '/deps/run.py', copy=True)
    .add_local_file(HERE / 'run.py.lock', '/deps/run.py.lock', copy=True)
    .run_commands('uv export --script /deps/run.py --frozen --no-hashes -o /deps/requirements.txt',
                  'uv pip install --system -r /deps/requirements.txt')
    .env({'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'OMP_NUM_THREADS': '4'})
    .add_local_dir(HERE, '/code', ignore=['__pycache__', '*.lock'])
)
app = modal.App('tandem-kv-translation', image=image)
runs = modal.Volume.from_name('tandem-runs', create_if_missing=True)
cache = modal.Volume.from_name('tandem-hf-cache', create_if_missing=True)


@app.function(gpu='H100', timeout=8 * 3600, secrets=[modal.Secret.from_name('huggingface')],
              volumes={str(RUNS): runs, '/root/.cache/huggingface': cache})
def job(args, commit, output, archive=True):
    """Run `python args...` in the code folder, then archive the output folder as experiments/<name>.

    run.py writes each run to <output>/<experiment>-<time>; that inner folder is archived, and `{run}` in
    args names it.
    """
    from huggingface_hub import HfApi
    os.environ['TANDEM_COMMIT'] = commit
    inner = lambda: [p for p in (RUNS / output).iterdir() if p.is_dir()]
    if any('{run}' in a for a in args):
        [run] = inner()
        args = [a.replace('{run}', str(run)) for a in args]
    try:
        subprocess.run(['python', *args], cwd='/code', check=True)
    finally:
        runs.commit()
        cache.commit()
    folder = RUNS / output
    if args[0] == 'run.py':
        [folder] = inner()
    if not archive:
        return folder.name, None, {}
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')}
    info = HfApi().upload_folder(repo_id=DATASET, repo_type='dataset', folder_path=folder,
                                 path_in_repo=f'experiments/{folder.name}',
                                 commit_message=f'Archive {folder.name} from {commit[:7]}')
    print(f'ARCHIVED experiments/{folder.name} at {info.oid}', digests)
    return folder.name, info.oid, digests


def clean_commit():
    status = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True, text=True)
    if status.stdout.strip():
        raise SystemExit('Commit the kv_translation changes first: results record the commit')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True,
                          check=True).stdout.strip()


@app.local_entrypoint()
def train_keys(steps: int = 4000, every: int = 500, validation: int = 100, archive: bool = True):
    """A smoke test: --steps 4 --every 2 --validation 2 --no-archive."""
    name = f'keyfit-{time.strftime("%Y%m%d-%H%M%S")}'
    args = ['train_keys.py', '--steps', str(steps), '--every', str(every), '--validation', str(validation),
            '--out', str(RUNS / name)]
    print(job.remote(args, clean_commit(), name, archive))


@app.local_entrypoint()
def experiment(name: str, count: int = 120, archive: bool = True):
    parent = f'{name}-{time.strftime("%Y%m%d-%H%M%S")}'
    print(job.remote(['run.py', 'run', name, '--count', str(count), '--out', str(RUNS / parent)],
                     clean_commit(), parent, archive))


@app.local_entrypoint()
def judge(run: str):
    """`run` is the output folder an experiment call used, such as key_fit-<time>."""
    print(job.remote(['run.py', 'judge', '{run}'], clean_commit(), run))
