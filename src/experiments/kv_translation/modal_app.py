"""Run KV translation jobs on Modal H100s and archive each result folder to the Tandem dataset.

Needs a Modal secret named `huggingface` with HF_TOKEN. The code must be committed: results record the commit.

  modal run --detach modal_app.py::train_keys --steps 4000             -> experiments/keyfit-<time>/
  modal run --detach modal_app.py::experiment --name bridge --count 500 --shards 5 --judge
                                                                        -> experiments/bridge-<time>/

An experiment splits its cases into consecutive shards, one H100 each. Each shard runs, then judges its own
answers, and `run.py merge` joins the shards into one run before it is archived.
"""
import hashlib
import math
import os
import subprocess
import sys
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
OPTIONS = dict(secrets=[modal.Secret.from_name('huggingface')],
               volumes={str(RUNS): runs, '/root/.cache/huggingface': cache})


def python(args, commit):
    subprocess.run([sys.executable, *map(str, args)], cwd='/code', check=True,
                   env=os.environ | {'TANDEM_COMMIT': commit})


def archive(folder, commit):
    from huggingface_hub import HfApi
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')}
    info = HfApi().upload_folder(repo_id=DATASET, repo_type='dataset', folder_path=folder,
                                 path_in_repo=f'experiments/{folder.name}',
                                 commit_message=f'Archive {folder.name} from {commit[:7]}')
    print(f'ARCHIVED experiments/{folder.name} at {info.oid}', digests, flush=True)
    return folder.name, info.oid, digests


@app.function(gpu='H100', timeout=8 * 3600, **OPTIONS)
def train(args, commit, folder, upload):
    try:
        python(['train_keys.py', *args, '--out', RUNS / folder], commit)
    finally:
        runs.commit()
        cache.commit()
    return archive(RUNS / folder, commit) if upload else (folder, None, {})


@app.function(timeout=3600, **OPTIONS)
def prefetch():
    """Put the pinned judge weights in the shared cache once, before shards read them."""
    from huggingface_hub import snapshot_download
    sys.path.insert(0, '/code')
    from judge import JUDGE
    snapshot_download(JUDGE[0], revision=JUDGE[1], allow_patterns=['*.json', '*.safetensors', '*.txt'])
    cache.commit()


@app.function(gpu='H100', timeout=8 * 3600, **OPTIONS)
def shard(name, start, count, commit, parent, judge):
    """Run one consecutive slice of an experiment, and judge it, in <parent>/shard-<start>/."""
    out = RUNS / parent / f'shard-{start:05d}'
    try:
        python(['run.py', 'run', name, '--start', start, '--count', count, '--out', out], commit)
        [run] = [p for p in out.iterdir() if p.is_dir()]
        if judge:
            python(['run.py', 'judge', run], commit)
    finally:
        runs.commit()  # The shard does not commit the cache: shards would race on the same files.
    return str(run)


@app.function(timeout=3600, **OPTIONS)
def merge(shards, commit, folder, upload):
    runs.reload()
    python(['run.py', 'merge', RUNS / folder, *shards], commit)
    runs.commit()
    return archive(RUNS / folder, commit) if upload else (folder, None, {})


def clean_commit():
    status = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True, text=True)
    if status.stdout.strip():
        raise SystemExit('Commit the kv_translation changes first: results record the commit')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True,
                          check=True).stdout.strip()


def stamp(name):
    return f'{name}-{time.strftime("%Y%m%d-%H%M%S")}'


@app.local_entrypoint()
def train_keys(steps: int = 4000, every: int = 500, validation: int = 100, upload: bool = True):
    """A smoke test: --steps 4 --every 2 --validation 2 --no-upload."""
    args = ['--steps', steps, '--every', every, '--validation', validation]
    print(train.remote([str(a) for a in args], clean_commit(), stamp('keyfit'), upload))


@app.local_entrypoint()
def experiment(name: str, count: int = 120, shards: int = 1, judge: bool = False, upload: bool = True):
    """A smoke test: --count 4 --shards 2 --judge --no-upload."""
    commit, folder = clean_commit(), stamp(name)
    if judge:
        prefetch.remote()
    size = math.ceil(count / shards)
    calls = [(name, start, min(size, count - start), commit, f'{folder}-shards', judge)
             for start in range(0, count, size)]
    print(merge.remote(list(shard.starmap(calls)), commit, folder, upload))
