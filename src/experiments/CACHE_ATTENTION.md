# Conditional cache attention

Translate SmolLM2-360M-Instruct caches into Qwen2.5-0.5B-Instruct caches and compare Qwen with native prefill on the same saved prompt. Both LLMs stay frozen. Training does not use answer labels or prompt reformatting. The checkpoint is one direction only.

| Script | Role |
| --- | --- |
| `train_cache_attention.py` | Fit the attention residual and its ridge control |
| `evaluate_cache_attention.py` | Greedy continuations, question replay, and K/V substitution |
| `calibrate_cache_attention.py` | Continue calibration on native Qwen generations |
| `run_continuation.sh`, `finish_continuation.sh` | Resume the Ada arms, then compare and upload them |
| `run_kv_ablation.sh` | Run the native K/V substitution on Ada |
| `cache_ridge.py`, `calibrate_cache_ridge.py` | Three-layer ridge map and its rank-16 correction |
| `scale_ridge_distillation.py` | 160-prompt versus 800-prompt distillation |
| `compare_ridge_layer_selection.py` | Fixed versus validation-selected source layers |
| `publish_ridge_distillation_data.py` | Publish that split |
| `evaluate_qa_answers.py` | BEM agreement and gold correctness on a saved report |
| `run_ridge_comparison.sh` | Three-layer ridge comparison on Ada or s4090a |

Run scripts with `uv run --locked --script`. Copied workers need `evaluate_cache_attention.py.lock`, the shared lock target. Generated tensors, reports, and credentials stay out of Git.

## Design

- One causal boundary at a time, using the manifest's character alignment. Attention is over source layers and heads, not text positions. Positions before the first causal match use a train-only target-mean prior.
- Source items are the 32 × 5 SmolLM layer/head vectors. Targets are the 24 × 2 Qwen layer/heads. K and V do not attend to each other. Embeddings cover model, layer, normalized depth, head, and role.
- One cross-attention block: width 64, four heads, FFN width 128, dropout zero, 48,768 trainable parameters. The output projection starts at zero, so the first prediction equals the ridge control.
- The control is a separate ridge map per target layer from the proportionally matched source layer and all of its KV heads. The intercept is unpenalized: 1,972,224 frozen coefficients.
- Mean and standard deviation are estimated on training prompts only, in float64. Standard deviation is clamped below 0.001. The loss is standardized K/V MSE, then native-teacher logit KL on the last four prompt tokens plus 0.01 standardized K/V anchoring.
- Predict pre-RoPE keys and projected values, then apply receiver RoPE once. This pair needs equal head dimension 64. Different head counts and depths are supported.

## Training

Use an allocated GPU, `CUDA_VISIBLE_DEVICES`, and a private Hugging Face login. The feature download is about 17.8 GB; allow at least 35 GB of server scratch.

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/pilot --data-root /path/to/shared-feature-cache \
  --pilot --epochs 1 --kl-steps 1 --upload-prefix experiments/your-unique-pilot
```

The pilot is four training and two validation prompts. Inspect `outputs/final.json`, `outputs/metrics.jsonl`, and peak GPU memory before the full run:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/full --data-root /path/to/shared-feature-cache \
  --epochs 20 --kl-steps 100 --hours 8 \
  --upload-prefix experiments/your-unique-full-run
```

Stride is eight receiver tokens and each batch has 16 aligned positions. Validation prompts are document-disjoint from the 160 training prompts. MSE stops after four unimproved validation epochs. KL calibration selects a separate checkpoint by validation logit KL. Neither selection uses answer labels. `--hours` is checked between epochs and KL steps. `--resume` continues the same work directory: MSE at epoch boundaries, KL at saved steps. Uploads happen after each MSE epoch, every 25 KL steps, and at completion. Local outputs stay if an upload fails. An existing remote prefix is refused. A lost work directory has to be restored from the cloud checkpoints by hand. Keep a separate test set before final claims.

`behavior-mse.json` is the comparison after reconstruction training. `final.json` is the ridge control against the selected attention checkpoint. Reports include teacher-forced suffix KL, top-1 agreement, native-replay maximum logit error, and peak PyTorch GPU allocation. Timing is one synchronized GPU measurement, not a GPU-to-GPU or cold-start measurement. Mapping time includes CPU-to-GPU copies, alignment, RoPE, cache construction, and the receiver suffix. It excludes source prefill, disk I/O, and tokenization.

## Free-running evaluation

`evaluate_cache_attention.py` loads the pinned behavior-selected checkpoint and compares up to 32 greedy tokens for native prefill, the ridge control, and attention. Saved native caches must match native logits and continuations on every prompt. Each path is timed five times after one warm-up. These validation prompts were used to select the checkpoint.

```sh
uv run --locked --script src/experiments/evaluate_cache_attention.py \
  --work /path/to/evaluation --data-root /path/to/shared-feature-cache \
  --device cuda --upload-prefix experiments/your-unique-evaluation
```

## Continuation calibration

`calibrate_cache_attention.py` starts from that checkpoint. Native Qwen generates up to 32 tokens. Logit KL supervises each generated position, with 0.01 standardized prefix K/V MSE. This is teacher-forced distillation on native continuations. `--pilot --epochs 2` uses eight training and four validation prompts. The full run allows 30 passes over 160 prompts and stops after five passes without validation KL improvement. `--tokens 1` is the one-token control. Both arms share the pass cap and patience, so the selected checkpoints can have different update counts. The checkpoint is selected by validation continuation KL; the parent is kept when calibration does not improve it. Eight training prompts are scored separately so fitting is not mistaken for generalization.

```sh
CUDA_VISIBLE_DEVICES=1 uv run --locked --script src/experiments/calibrate_cache_attention.py \
  --work /path/to/new-pilot --data-root /path/to/existing-feature-cache \
  --pilot --epochs 2 --upload-prefix experiments/your-unique-continuation-pilot
```

Native cache replay must match before training. Checkpoints save every 25 steps and each epoch. Uploads are at most once every ten minutes, plus a final attempt. HTTP 429 waits at least an hour and honors a longer `Retry-After`. `upload-pending.json` records the retry. `--resume --epochs 30 --patience 5` restores weights, Adam state, RNG state, shuffle position, and early stopping. Model, dataset, split, continuation length, seed, and learning rate must match, and native replay is checked again. An interruption can drop up to 24 unsaved updates. An older checkpoint that omitted Adam state keeps its weights, resets Adam once, and records that reset. A work-directory lock blocks a second launch.

`run_continuation.sh` checks `continuation.SHA256SUMS`, skips arms that already finished or early-stopped, refuses to start while your calibration process is running, and gives each arm two hours on Ada GPU 0. `finish_continuation.sh` evaluates both selected checkpoints for 32 greedy tokens on the same 40 prompts, with one timing repeat, requires identical native continuations, and uploads the two training folders and two evaluation reports. Rerunning it reuses saved evaluations. Keep the Ada cache until all four uploads are verified.

The [verified reports](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/ba8bfb17f27e5e6e10571093814c75a0c1fbef5c/experiments):

| Calibration | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: |
| One token | 22/40 | 0.625 tokens | 0/40 |
| 32 tokens | 19/40 | 1.225 tokens | 0/40 |

Native replay passed. The 32-token model keeps the topic and changes names: native “Pam Veasey and Jon Jost” becomes “Pam Veasey and Jon Jorsky.” Two prompts match at least five tokens; none match ten. Teacher-forced agreement of 64.6% does not establish independent generation retention. Eight monitored training prompts reach KL 0.613 and a 6.25-token mean prefix, versus KL 1.366 and 1.225 tokens on development prompts. That subset is not a full training-set estimate. Validation KL is best at epoch 23 and worse through epoch 28 while training loss falls. Exact character-end alignment covers 97.1% of development prompt tokens and does not show that the mapped features are accurate.

## Question replay

`--handoff question` translates the prefix before the question, then lets Qwen read the unchanged question and `Answer:` suffix (10–45 tokens here). That is partial receiver prefill. Use the same 32-token checkpoint as `--handoff last-token`. The comparison needs no new training or shards.

```sh
for handoff in last-token question; do
  uv run --locked --script src/experiments/evaluate_cache_attention.py \
    --checkpoint-file /path/to/full-tokens-32/outputs/best.pt \
    --work /path/to/new-diagnostic/$handoff --data-root /path/to/existing-feature-cache \
    --handoff "$handoff" --tokens 32 --limit 40 --repeats 1
done
```

The [Ada comparison](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/8e1f5515e2a5be16531a210a756f35dc70210ca5/experiments/question-replay-20261029-205336) uses the same checkpoint hash, prompt IDs, and native continuations. Native replay passed.

| Attention mapper handoff | First-token KL | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: | ---: |
| Final prompt token | 2.156 | 19/40 | 1.225 tokens | 0/40 |
| Question and Answer: suffix | 1.600 | 16/40 | 3.275 tokens | 0/40 |

Question replay lowers first-token KL on 31/40 prompts. Prefixes improve on 10, worsen on 8, and stay the same on 22. Eight retain at least five tokens. The boxer example keeps “Tumba Silva” from the question and replaces native “Clemente Russo” with “Sven Mikkelsen.” Exact match also penalizes rephrasing: native “Afghanistan.” becomes “The country where Mashorey is located is Afghanistan.” This is a partial benefit. It does not preserve behavior reliably, and it does not isolate a layer or K/V cause. No extra training ran.

## Native K/V substitution

`--kv-ablation --handoff last-token` adds native keys with translated values, and translated keys with native values, in pre-RoPE space. The substitutions need native receiver caches. They are not a timing or prefill-saving result. `run_kv_ablation.sh <full-git-commit>` checks the saved checkpoint and bundle hashes, refuses an occupied GPU or a duplicate job, and uploads one report. An existing report is kept. Run it under `nohup`. If the upload fails, the local report remains.

The [Ada report](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/blob/f277905af032112cc9bc315e9a7fef4eae35609a/experiments/kv-ablation-1641c9063a85/evaluation.json) reproduces the translated baseline. Native replay passed.

| Prefix cache | First-token KL | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: | ---: |
| Translated K and V | 2.156 | 19/40 | 1.225 tokens | 0/40 |
| Native K, translated V | 2.003 | 18/40 | 0.825 tokens | 0/40 |
| Translated K, native V | 1.623 | 22/40 | 2.425 tokens | 0/40 |

Replacing values lowers first-token KL on 34/40 prompts, versus 22/40 for keys. With native values, first-token matches gain eight prompts and lose five; prefixes improve on 12 and worsen on nine. Value translation is the stronger lead. Keys and values still interact, so this is not an additive split, and neither substitution restores a full continuation.

## Ridge experiments

`calibrate_cache_ridge.py` fits a rank-16 correction using receiver attention at question tokens. `scale_ridge_distillation.py` compares five passes over 160 prompts with one pass over 800 prompts, both for 800 updates. On 120 held-out questions, the larger split moves first-token KL from 1.85 to 1.57 and answer F1 from 9.4% to 9.8%.

Load the [Distilled 800 checkpoint](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/blob/f4383463b411213110446e03d1558a908677492a/experiments/ridge-distillation-800-20261111/distilled_800.pt) by initializing `ThreeLayerRidge` from the [parent normalization checkpoint](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/blob/ba8bfb17f27e5e6e10571093814c75a0c1fbef5c/experiments/continuation-ada-full-tokens-32-20261029/best.pt), as in `scale_ridge_distillation.py`, then assign the saved `weights` tensor to `ridge.weights`.

`compare_ridge_layer_selection.py` fits three maps on the original 160 prompts and ranks layers on the 40 development prompts. On the same 120 test prompts, twelve selected layers improve 16-token passage recovery from 76.0% to 83.7% over fixed ridge, using 23.6M coefficients versus 5.9M. Answer F1 is 4.1% versus 3.2%. Native Qwen scores 15.0% F1. Passage recovery is the share of the loss gap between no context and native prefill that the translated cache closes. The script reads `test.parquet` and the adjacent `manifest.json`.

## Answer evaluation

```sh
uv run --locked --script src/experiments/evaluate_qa_answers.py \
  /path/to/evaluation.json --method selected_12 --output /path/to/semantic.json
```

Use `--method distilled_800` for the 800-prompt map. Older distillation reports need `--test-data /path/to/test.parquet` to recover the question. Older layer-selection reports omit responses and have to be regenerated.

The scorer is the pinned [BEM reproduction](https://huggingface.co/kortukov/answer-equivalence-bem). Agreement requires at least 0.5 in both directions. Gold correctness scores each response against the gold answer. The report counts preserved, lost, and corrected answers. BEM was trained on SQuAD judgments, so check verbose, contradictory, and yes/no HotpotQA answers before using it as a gate.
