#!/usr/bin/env bash
set -euo pipefail
test "$(hostname -s)" = ada || { echo "Connect to Ada first."; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
umask 077
if pgrep -u "$(id -u)" -f '[c]alibrate_cache_attention.py|[e]valuate_cache_attention.py|[f]inish_continuation.py' >/dev/null; then
  echo "Your training or evaluation is still running. Stopping this launcher."
  exit 1
fi
run_home="$HOME/tandem-cluster/continuation-ada"
scratch=$(cat "$run_home/scratch-path.txt")
test -d "$scratch/data" || { echo "Ada feature cache is missing."; exit 1; }
export UV_CACHE_DIR="$scratch/uv"
export UV_PYTHON_INSTALL_DIR="$scratch/python"
export HF_HUB_CACHE="$scratch/hub"
export HF_XET_CACHE="$scratch/xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ ! -f "$run_home/evaluation-32-tokens-1/evaluation.json" ] ||
   [ ! -f "$run_home/evaluation-32-tokens-32/evaluation.json" ]; then
  test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 512 || {
    echo "GPU 0 is occupied. Stopping."
    exit 1
  }
fi
cd "$(dirname "$0")"
if [[ -f SHA256SUMS ]]; then
  sha256sum -c SHA256SUMS
else
  sha256sum -c continuation.SHA256SUMS
fi
CUDA_VISIBLE_DEVICES=0 exec uv run --locked --script finish_continuation.py --run-home "$run_home"
