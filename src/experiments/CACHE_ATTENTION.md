# Conditional cache attention: first experiment

Goal: translate SmolLM2-360M-Instruct caches into Qwen2.5-0.5B-Instruct caches and preserve Qwen's native behavior on the same saved prompt. No answer labels or prompt reformatting are used. Both LLMs remain frozen. The first checkpoint is trained in one direction; conditioning on model IDs does not imply support for unseen models or pairs.

## Design decisions

- **One causal boundary at a time.** Align using the existing manifest's character-boundary correspondence. Attention runs over source layers/heads, not across text positions. A future source state cannot help predict an earlier receiver state. Positions before the first causal match are excluded from fitting and use a train-only target-mean prior during replay.
- **Separate K/V observations, shared weights.** For each role and text position, source items are the 32 × 5 SmolLM layer/head vectors. Target queries identify the 24 × 2 Qwen layer/heads. Model, layer, normalized depth, head and role embeddings identify each item/query. K and V never attend to each other's observations.
- **Small attention residual.** One cross-attention block, width 64, four heads, FFN width 128, dropout zero. The 48,768 trainable parameters predict a correction to an affine baseline; the output projection starts at zero, so initial predictions exactly equal that baseline. These are architecture choices for this experiment, not settings validated by a prior paper.
- **Affine control.** Fit separate ridge maps for K/V at each target layer, using the proportionally matched source layer and all its KV heads. The intercept is unpenalized. This control has 1,972,224 frozen coefficients and is not a reproduction of HeteroFold's Recolor or multi-layer ridge methods.
- **Train-only scaling.** Estimate mean/std per feature using training prompts only, in float64. Clamp standard deviation below 0.001. Optimize standardized K/V MSE so high-magnitude heads/layers do not dominate by scale alone.
- **Receiver behavior matters.** After MSE training, calibrate the attention residual with native-teacher-to-translated-student logit KL plus 0.01 standardized K/V anchoring. Both receiver weights and native targets remain frozen. Four final tokens of each prompt are teacher-forced as a suffix; only the preceding cache is translated. This is a bounded logit-distillation experiment, not HeteroFold's attention/output objective.
- **Inference conventions.** Predict pre-RoPE keys and projected values, then apply receiver RoPE once at receiver positions. Target queries in the translator come from metadata and affine predictions, not receiver-prefilled context. The source LLM need not be resident during training on saved features.

The implementation currently requires the two supported architectures and equal head dimensions (64 for this pair). Different head counts and depths are supported. Arbitrary architecture support would require more extractors, positional handling, input/output adapters and multi-pair training.

## Evidence informing the design

[HeteroFold, §§3 and B.2](https://arxiv.org/html/2609.32259v2) motivates causal token alignment, content-space keys and receiver-aware supervision: low K/V reconstruction error alone is insufficient. Its corrections fold into affine maps; our attention network does not, so its runtime must be measured.

[Mixture-of-Translators, Appendix C.3](https://arxiv.org/html/2607.28979v2#A3.SS3) provides precedent for cross-attention cache translation with separate K/V processing. Its recurrent channel-window translators and expert routing differ from this per-boundary shared network; its results do not validate our model-ID conditioning scheme.

[Heo et al.](https://arxiv.org/abs/2608.03893) report pair-dependent linear-transfer quality and cases where nonlinear MLPs help. Their matched-KV, within-family experiments are not a guarantee for Qwen/SmolLM.

Community implementation evidence is mixed. [spotta85/kvbridge](https://github.com/spotta85/kvbridge) reports quality/runtime tradeoffs and distinguishes warm handoff from cold execution. [souvikDevloper/kvbridge](https://github.com/souvikDevloper/kvbridge) explicitly reports a real T4 run failing its quality gates despite faster mapping. These are authors' reported results, not our reproductions. They support including a baseline, replay control, behavior metrics and explicit timing boundaries.

## Cluster pilot, then full run

Use an allocated GPU via `CUDA_VISIBLE_DEVICES`. Authenticate to Hugging Face on the server. The full feature download is approximately 17.8 GB, plus CUDA dependencies, receiver weights and checkpoints; allow at least 35 GB of server scratch. Bulk tensors never need to be on the laptop.

First run four training and two validation prompts, one MSE epoch and one KL step:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/pilot --data-root /path/to/shared-feature-cache \
  --pilot --epochs 1 --kl-steps 1 --upload-prefix experiments/your-unique-pilot
```

The pilot must finish native replay controls and backward passes without errors. Inspect `outputs/final.json`, `outputs/metrics.jsonl` and peak allocated GPU memory before launching the larger run. The measured epoch time provides a first runtime estimate; four-prompt throughput must be scaled to 160 training / 40 validation prompts, and download, fitting, receiver calibration and evaluation add time. Overnight is a budget, not a convergence guarantee.

Then use the same feature cache and a separate output directory/prefix:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/full --data-root /path/to/shared-feature-cache \
  --epochs 20 --kl-steps 100 --hours 8 \
  --upload-prefix experiments/your-unique-full-run
```

The training stride is eight receiver tokens and batches contain 16 aligned positions. All source layer/head features at each selected boundary are retained. Validation prompts are document-disjoint from training. MSE training stops after four unimproved validation epochs; select the MSE checkpoint by validation loss. KL calibration selects a separate checkpoint by validation logit KL. Neither selection uses answer labels. Keep a separate test set before final claims.

`--hours` is a soft training budget checked between epochs and KL steps; preparation and final evaluation can extend total wall time. To resume, retain the work directory, feature cache and identical configuration, adding `--resume`. MSE resumes at epoch boundaries and KL resumes at saved step boundaries. With `--upload-prefix`, verified checkpoints are uploaded after each MSE epoch, every 25 KL steps and at completion. Local outputs are retained on upload failure. Initial runs refuse an already-used remote prefix. Restoring a lost work directory solely from cloud checkpoints requires manually downloading the outputs first.

## Reading results

`behavior-mse.json` compares both methods after reconstruction training. `final.json` compares the affine control with the selected attention checkpoint after calibration. Reports include teacher-forced suffix logit KL, top-1 agreement, the native replay control's maximum logit error, and peak PyTorch GPU allocation.

Timing is exploratory, single-GPU wall time with CUDA synchronization. Mapping measurements include CPU-source-feature copies, alignment, mapping, RoPE/cache construction and receiver suffix processing. Native timing includes receiver prefill and the same suffix. Source prefill, file I/O and tokenization are excluded. Single measurements are not sufficient to claim a production speedup; use matched warm-ups and repeated measurements before any latency claim. No full-text answer accuracy, free-running generation retention, multi-turn stability or universal-translation result is established by this experiment.

## Free-running evaluation

`evaluate_cache_attention.py` loads the pinned behavior-selected checkpoint and the same validation prompts. It compares up to 32 independently generated greedy tokens for native prefill, affine transfer and attention transfer, recording first-token agreement, matching prefix length and exact sequence agreement. It checks native saved-cache logits and continuations against native prefill for every prompt. Outputs include generated text but no answer labels.

It also reports five synchronized timings per prompt after a warm-up for each path. Native prefill and mapped handoff run on one GPU; mapping includes CPU-to-GPU source-cache copies, alignment, RoPE/cache construction and first-token computation, but excludes source prefill, disk I/O and tokenization. These are not GPU-to-GPU or cold-start measurements. Validation was used for checkpoint selection; this is development evaluation, not a final held-out test.

```sh
uv run --locked --script src/experiments/evaluate_cache_attention.py \
  --work /path/to/evaluation --data-root /path/to/shared-feature-cache \
  --device cuda --upload-prefix experiments/your-unique-evaluation
```

## Continuation calibration

`calibrate_cache_attention.py` starts from the pinned behavior-selected checkpoint and reuses the existing feature cache. Native Qwen generates up to 32 tokens per prompt. The translated path receives the final prompt token and the preceding native continuation tokens; logit KL supervises each generated position, with 0.01 standardized prefix K/V MSE anchoring. Both LLMs stay frozen. This is teacher-forced distillation on native continuations; independent greedy generation is measured before and after training.

Use `--pilot --epochs 2` for eight training and four validation prompts. Inspect `outputs/comparison.json` and measured GPU memory before expanding. The full run allows up to 30 shuffled passes over all 160 training prompts, stopping after five complete passes without validation KL improvement. `--tokens 1` provides the one-token calibration control; both arms use the same maximum passes and patience, so their selected checkpoints may use different update counts. Select the checkpoint by validation continuation KL, retaining the parent if calibration does not improve it. Reports also compare generation on eight training prompts to distinguish fitting from generalization. Existing validation remains development data.

```sh
CUDA_VISIBLE_DEVICES=1 uv run --locked --script src/experiments/calibrate_cache_attention.py \
  --work /path/to/new-pilot --data-root /path/to/existing-feature-cache \
  --pilot --epochs 2 --upload-prefix experiments/your-unique-continuation-pilot
```

Native cache logits and greedy continuations must match native prefill before training. Outputs include trace token IDs, parent checkpoint identity, configuration, metrics and remotely verified checkpoints. Local checkpoints save every 25 steps and each epoch. Cloud uploads occur at most once every ten minutes during training, plus a final attempt. Rate limits and transient network/server failures defer publication without stopping training. A persistent `upload-pending.json` records the retry time; HTTP 429 waits at least an hour, honoring a longer `Retry-After`. Pending final outputs stay on the server for publication after the cooldown. A work-directory lock prevents duplicate launches. The soft time budget is checked between training prompts and excludes final evaluation time.

To continue, keep the work directory, feature cache and upload prefix, then add `--resume --epochs 30 --patience 5`. Epochs are the total cap, including completed passes. Resume restores mapper weights, Adam state, RNG state, shuffle position and early-stopping history from the latest checkpoint; an abrupt interruption can lose up to 24 unsaved updates. Model, dataset, split, continuation length, seed and learning rate must match. Native replay is rechecked before continuing. The first version omitted Adam state; importing its checkpoint preserves weights and update counts, resets Adam once and records the reset in resume provenance. Cached traces and original baselines are reused.

`run_continuation.sh` is the Ada launcher bundled with these scripts and their checksums. After the original comparison finishes, it continues each existing arm toward 30 total passes with patience five, or initializes an arm that has not started. Completed and early-stopped arms are skipped. It refuses to launch while another calibration process owned by the user is running. Each invocation allows two hours of training per arm; rerun the launcher if that budget is reached before convergence.

`finish_continuation.sh` evaluates both selected checkpoints on the same 40 development prompts, generating up to 32 greedy tokens per prompt. It uses `evaluate_cache_attention.py --checkpoint-file` with native replay controls and records each checkpoint's SHA-256. Both reports must contain identical native continuations before comparison. One timing repeat per path is included for diagnostics; this run does not establish a speedup.

After evaluation, the finish script uploads and verifies both training-output folders and both evaluation reports, using at most four folder commits. Deferred uploads keep their files on Ada. Rerunning the script reuses matching saved evaluations and retries publication without retraining or regenerating continuations. Keep the Ada cache and output directories until all four uploads are verified.

## Continuation findings and question-replay diagnostic

The [verified reports](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/ba8bfb17f27e5e6e10571093814c75a0c1fbef5c/experiments) compare the selected checkpoints on the same 40 development prompts, with up to 32 greedy tokens:

| Calibration | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: |
| One token | 22/40 | 0.625 tokens | 0/40 |
| 32 tokens | 19/40 | 1.225 tokens | 0/40 |

All native replay controls passed. The 32-token model often retains the topic but changes names or details: native “Pam Veasey and Jon Jost” becomes “Pam Veasey and Jon Jorsky.” Only two prompts match at least five leading tokens; none match ten. Teacher-forced agreement of 64.6% therefore does not establish independent generation retention.

The eight monitored training prompts reach KL 0.613 and a 6.25-token mean matching prefix, versus KL 1.366 and 1.225 tokens on development prompts. This small training subset suggests a generalization gap; it is not a full training-set estimate. Validation KL is best at epoch 23 and then worsens through epoch 28 while training loss falls. More passes are not the priority. Alignment has exact character-end matches for 97.1% of development prompt tokens, but that does not establish accurate mapped features.

Compare `--handoff last-token` with `--handoff question` using the **same 32-token checkpoint**. The latter translates the prefix before the first question token, then lets Qwen process the unchanged question and `Answer:` suffix: 10–45 tokens on these prompts. This tests whether recomputing the question helps Qwen use the translated context. It introduces partial receiver prefill and is not HeteroFold's prefill-free method. Neither outcome alone localizes a layer, K/V component, or architectural cause.

```sh
for handoff in last-token question; do
  uv run --locked --script src/experiments/evaluate_cache_attention.py \
    --checkpoint-file /path/to/full-tokens-32/outputs/best.pt \
    --work /path/to/new-diagnostic/$handoff --data-root /path/to/existing-feature-cache \
    --handoff "$handoff" --tokens 32 --limit 40 --repeats 1
done
```

Run on one allocated GPU with `CUDA_VISIBLE_DEVICES` set. Each report records the checkpoint hash, handoff mode, replayed token count, first-token KL and rank of the native first token, plus the existing generation metrics. Native saved-cache logits and continuations must match full prefill at each handoff boundary. Compare identical prompt IDs and native continuations across the two reports. This diagnostic needs no training, new shards or answer labels.

The [completed Ada comparison](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/8e1f5515e2a5be16531a210a756f35dc70210ca5/experiments/question-replay-20261029-205336) uses identical checkpoint hashes, prompt IDs and native continuations. All native replay controls pass.

| Attention mapper handoff | First-token KL | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: | ---: |
| Final prompt token | 2.156 | 19/40 | 1.225 tokens | 0/40 |
| Question and Answer: suffix | 1.600 | 16/40 | 3.275 tokens | 0/40 |

Question replay lowers first-token KL on 31/40 prompts. Matching prefixes improve on 10 prompts, worsen on eight and remain unchanged on 22; eight now retain at least five leading tokens. Qualitative inspection shows better retention of names and wording supplied in the question, but passage details still change. For example, the boxer example retains “Tumba Silva” from the question while replacing native “Clemente Russo” with “Sven Mikkelsen.” Exact matching also penalizes harmless rephrasing: native “Afghanistan.” becomes “The country where Mashorey is located is Afghanistan.” These results support a partial benefit, not reliable behavior preservation or an identified architectural cause. No additional training ran.

## Native K/V substitution diagnostic

Add `--kv-ablation --handoff last-token` to evaluate the selected 32-token checkpoint on the same 40 prompts with two extra conditions: native keys plus translated values, and translated keys plus native values. Replace only the chosen prefix component in pre-RoPE feature space; apply receiver RoPE once and generate normally after handoff. The existing native replay control, translated attention path and affine control run alongside them. Compare first-token KL, first-token agreement and independent continuations within the same report.

If native keys improve retention, translated keys deserve further investigation; if native values help, investigate value translation. Neither intervention isolates an additive error contribution: keys and values interact through the receiver, and a hybrid cache can differ from both native and jointly translated caches. These are diagnostic substitutions requiring native receiver caches, with no deployment timing or prefill-saving claim. No retraining is involved.

`run_kv_ablation.sh <full-git-commit>` prepares and runs this comparison on Ada GPU 0. It verifies the saved checkpoint, downloads the evaluator and dependencies including its lockfile, checks the bundle hashes, rejects an occupied GPU or duplicate job, and uploads one report. Run it under `nohup` to survive disconnects. An existing report is retained and never regenerated automatically; if upload fails, the log identifies the failure and the local report remains available for publication.

The [completed Ada report](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/blob/f277905af032112cc9bc315e9a7fef4eae35609a/experiments/kv-ablation-1641c9063a85/evaluation.json) uses the same selected checkpoint and 40 development prompts. Native replay controls pass, and the fully translated baseline reproduces the preceding evaluation.

| Prefix cache | First-token KL | First token matches | Mean matching prefix | Full continuation matches |
| --- | ---: | ---: | ---: | ---: |
| Translated K and V | 2.156 | 19/40 | 1.225 tokens | 0/40 |
| Native K, translated V | 2.003 | 18/40 | 0.825 tokens | 0/40 |
| Translated K, native V | 1.623 | 22/40 | 2.425 tokens | 0/40 |

Replacing values lowers first-token KL on 34/40 prompts, versus 22/40 for replacing keys. With native values, first-token matches gain eight prompts and lose five; matching prefixes improve on 12 and worsen on nine. The evidence favors investigating value translation next, but does not establish that keys are adequate or that a particular value-training change will help. Neither substitution restores reliable independent generation, and exact matching still penalizes some harmless rephrasing.

## Ridge experiments

`calibrate_cache_ridge.py` fits a rank-16 correction to the three-layer ridge map using receiver attention at question tokens. `scale_ridge_distillation.py` starts from its plain ridge map and compares five passes over 160 prompts with one pass over 800 prompts. Both use 800 updates on native Qwen continuations. On 120 held-out questions, expanding the split lowered first-token KL from 1.85 to 1.57 and moved answer F1 from 9.4% to 9.8%.

`compare_ridge_layer_selection.py` fits three maps on the original 160 prompts and ranks candidate layers on the 40 development prompts. It compares three fixed layers, three selected layers, and twelve selected layers on the same 120 test prompts. The twelve-layer map improved 16-token passage recovery from 76.0% to 83.7% over fixed ridge. It used 23.6M coefficients versus 5.9M and scored 4.1% answer F1 versus 3.2%. Native Qwen scored 15.0% F1. Passage recovery measures the share of the prediction loss gap between no context and native prefill that a translated cache closes.

Run the ridge scripts with `uv run --locked --script` and `--help` for their inputs. Layer selection takes the published `test.parquet` with its adjacent `manifest.json`. Keep the shared script lock target `evaluate_cache_attention.py.lock` with the scripts when copying them to a worker.

## Answer evaluation

The ridge evaluations save the question, gold answer, and complete generated responses alongside first-line EM/F1 and token diagnostics. Score the saved responses on CPU:

```sh
uv run --locked --script src/experiments/evaluate_qa_answers.py \
  /path/to/evaluation.json --method selected_12 --output /path/to/semantic.json
```

Use `--method distilled_800` for the 800-prompt map. For older distillation reports, pass `--test-data /path/to/test.parquet` to recover question text by ID. Older layer-selection reports require regeneration because they omit responses.

The scorer uses a pinned [BEM reproduction](https://huggingface.co/kortukov/answer-equivalence-bem) locally. Agreement requires a score of at least 0.5 in both directions. Gold correctness uses the gold answer as reference and scores each response independently. The report counts preserved, lost, and corrected answers and includes each response and score for review. BEM was trained on SQuAD answer judgments. Check its decisions on verbose, contradictory, and yes/no HotpotQA answers before using it as a quality gate.
