#!/usr/bin/env bash
# Compare the saved one-layer ridge and attention with a matched three-layer fit.
set -euo pipefail
host=$(hostname -s)
case "$host" in ada|s4090a) ;; *) echo "Run this on Ada or s4090a."; exit 1 ;; esac
gpu=${1:?Pass the GPU index shown by nvidia-smi}
export PATH="$HOME/.local/bin:$PATH"
umask 077

root="$HOME/tandem-cluster/continuation-ada"
checkpoint="$root/full-tokens-32/outputs/best.pt"
printf '713c5c140c01d42b26514025c4624a74d4493fa175243e4eb202a30c852371ad  %s\n' "$checkpoint" | sha256sum -c -
cd "$(dirname "$0")"
sha256sum -c continuation.SHA256SUMS

exec 9>"$root/ridge-comparison.lock"
flock -n 9 || { echo "A ridge comparison is already running."; exit 1; }

work=$(mktemp -d "$root/ridge-comparison.XXXXXX")
echo "Results directory: $work"
scratch=${2:-}
if [[ -z "$scratch" && "$host" = ada && -f "$root/scratch-path.txt" ]]; then
  scratch=$(cat "$root/scratch-path.txt")
fi
if [[ ! -d "$scratch/data" ]]; then
  scratch=$(mktemp -d "/tmp/tandem-$(id -un)-ridge.XXXXXX")
  mkdir "$scratch/data"
fi
printf '%s\n' "$scratch" > "$work/scratch-path.txt"
export UV_CACHE_DIR="$scratch/uv"
export UV_PYTHON_INSTALL_DIR="$scratch/python"
export HF_HUB_CACHE="$scratch/hub"
export HF_XET_CACHE="$scratch/xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

{
  echo "Checking pinned dataset files on $host"
  uvx --from huggingface-hub==0.36.0 hf download tandemkv/paired-kv-qwen-smollm \
    --repo-type dataset --revision dca861cd61b10859d5682565cfbba5606e5c5974 \
    --include 'runs/hotpot-200-20261028/*' --local-dir "$scratch/data"

  IFS=, read -r uuid used utilization < <(nvidia-smi -i "$gpu" \
    --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  [[ "${used// /}" =~ ^[0-9]+$ && "$used" -lt 512 && "$utilization" -eq 0 ]] || {
    echo "GPU $gpu is occupied or unavailable. Reuse the staged data at $scratch."
    exit 1
  }
  CUDA_VISIBLE_DEVICES="$uuid" uv run --locked --script evaluate_cache_attention.py \
    --checkpoint-file "$checkpoint" --data-root "$scratch/data" \
    --work "$work" --compare-ridge-layers --handoff last-token \
    --tokens 32 --limit 40 --repeats 5
} 2>&1 | tee "$work/run.log"
