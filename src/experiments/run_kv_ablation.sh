#!/usr/bin/env bash
# Run the existing Ada checkpoint's native-component diagnostic, without training.
set -euo pipefail
test "$(hostname -s)" = ada || { echo "Connect to Ada first."; exit 1; }
revision=${1:?Pass the full Git commit containing this launcher}
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full Git commit."; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
umask 077

root="$HOME/tandem-cluster/continuation-ada"
exec 9>"$root/kv-ablation.lock"
flock -n 9 || { echo "Your K/V diagnostic is already running."; exit 1; }
if pgrep -u "$(id -u)" -f '[c]alibrate_cache_attention.py|[e]valuate_cache_attention.py|[f]inish_continuation.py' >/dev/null; then
  echo "Your training or evaluation is still running. Stopping."
  exit 1
fi
scratch=$(cat "$root/scratch-path.txt")
test -d "$scratch/data" || { echo "Existing Ada data cache is missing."; exit 1; }
checkpoint="$root/full-tokens-32/outputs/best.pt"
printf '713c5c140c01d42b26514025c4624a74d4493fa175243e4eb202a30c852371ad  %s\n' "$checkpoint" | sha256sum -c -
run_id="kv-ablation-${revision:0:12}"
work="$root/$run_id"
if [ -f "$work/evaluation/evaluation.json" ]; then
  echo "Saved evaluation already exists: $work/evaluation/evaluation.json"
  echo "Check the previous log for upload confirmation. No evaluation restarted."
  exit 0
fi
mkdir -p "$work/bundle"
for file in cache_attention.py train_cache_attention.py paired_kv_dataset.py evaluate_cache_attention.py evaluate_cache_attention.py.lock continuation.SHA256SUMS; do
  curl -fsSL --retry 3 "https://raw.githubusercontent.com/tandemkv/Tandem/$revision/src/experiments/$file" -o "$work/bundle/$file"
done
cd "$work/bundle"
sha256sum --ignore-missing -c continuation.SHA256SUMS

export UV_CACHE_DIR="$scratch/uv"
export UV_PYTHON_INSTALL_DIR="$scratch/python"
export HF_HUB_CACHE="$scratch/hub"
export HF_XET_CACHE="$scratch/xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 512 || {
  echo "GPU 0 is occupied. Stopping."
  exit 1
}
CUDA_VISIBLE_DEVICES=0 uv run --locked --script evaluate_cache_attention.py \
  --checkpoint-file "$checkpoint" --data-root "$scratch/data" \
  --work "$work/evaluation" --kv-ablation --handoff last-token \
  --tokens 32 --limit 40 --repeats 1 --upload-prefix "experiments/$run_id"
