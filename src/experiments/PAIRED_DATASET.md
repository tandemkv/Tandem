# Paired KV dataset

The [private Hugging Face dataset](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm) contains passages and questions from 200 HotpotQA examples and paired model features. Its card describes the data, licensing and split. Gold answer labels are excluded. The feature manifest is immutable because each shard records its hash.

`paired_kv_dataset.py` prepares complete prompts, token boundaries and splits; extracts frozen model key projections before RoPE and value projections; and verifies local shards. `cloud_prefill.py` uses that extractor to upload and verify batches before deleting temporary copies. `replay_batch.py` downloads one shard at a time and runs `replay_cache.py` to compare saved-cache next-token logits against native prefill.

The scripts pin the source and model revisions. Use Python through `uv`; do not put model weights, tokens, output features, reports or credentials in Git. Log in with `uvx hf auth login` on the machine running the scripts. The runner refuses public destination datasets. A complete CUDA job requires server storage for dependencies, models, the source dataset and temporary batches; check your allocation and quota first.

Example, from the repository root, using a GPU allocated to you:

```sh
export CUDA_VISIBLE_DEVICES=1
uv run --locked --script src/experiments/cloud_prefill.py \
  --repo tandemkv/paired-kv-qwen-smollm \
  --prefix runs/your-unique-run --work /path/to/private/scratch \
  --count 200 --batch 5
```

Select a unique prefix and an empty work directory. To resume after an interruption, use the same arguments and work directory. The runner verifies uploaded file hashes at each checkpoint before removing that batch. `complete.json` records completion; `last_upload.json` records the corresponding immutable revision. Do not run two processes against the same prefix.

For the existing 200-prompt dataset, the replay check covers five training and five validation prompts sampled across lengths for both models:

```sh
uv run --locked --script src/experiments/replay_batch.py --work /path/to/private/scratch --device cuda
```

The next experiment fits a mapper on training features, then compares the receiver's logits and predictions from a translated cache against native receiver prefill on the same validation prompt. The goal is to preserve native behavior; gold-answer QA accuracy is outside this initial check. The completed replay tests cover same-model cache restoration, not cross-model transfer. This is an initial HotpotQA-only dataset, not HeteroFold's full calibration corpus.
