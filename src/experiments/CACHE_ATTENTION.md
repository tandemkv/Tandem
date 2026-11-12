# Conditional cache attention

Translate SmolLM2-360M-Instruct caches into Qwen2.5-0.5B-Instruct caches and compare Qwen's greedy continuation with native prefill on the same saved prompt. Both LLMs stay frozen. Gold answers are not used for training.

The translator is one 64-wide, four-head cross-attention block over a frozen per-layer ridge control. It predicts pre-RoPE keys and projected values at one causal boundary at a time. K and V do not attend to each other. The output projection starts at zero, so the first prediction equals the ridge control. Train on standardized K/V MSE, then calibrate with native-teacher logit KL on the last four prompt tokens plus 0.01 K/V anchoring.

Use a GPU you were allocated, `CUDA_VISIBLE_DEVICES`, and a private Hugging Face login. Feature shards stay in the private dataset; do not commit weights, tensors, or reports.

Pilot, then the full run, each with its own work directory and upload prefix:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/pilot --data-root /path/to/shared-feature-cache \
  --pilot --epochs 1 --kl-steps 1 --upload-prefix experiments/your-unique-pilot

uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/full --data-root /path/to/shared-feature-cache \
  --epochs 20 --kl-steps 100 --hours 8 \
  --upload-prefix experiments/your-unique-full-run
```

`--hours` is checked between epochs and KL steps. Add `--resume` to continue the same work directory. The runner refuses an upload prefix that already exists.

Free-running evaluation loads the behavior-selected checkpoint and compares up to 32 greedy tokens for native prefill, the ridge control, and attention. It also checks that saved native caches match native prefill.

```sh
uv run --locked --script src/experiments/evaluate_cache_attention.py \
  --work /path/to/evaluation --data-root /path/to/shared-feature-cache \
  --device cuda --upload-prefix experiments/your-unique-evaluation
```

`--handoff question` lets Qwen reread the question over a translated prefix. `--kv-ablation` swaps native keys or values into the translated cache. Both are diagnostics. These validation prompts were used to select the checkpoint.

Saved generations can be scored for agreement and gold correctness on CPU:

```sh
uv run --locked --script src/experiments/evaluate_qa_answers.py \
  /path/to/evaluation.json --method attention --output /path/to/semantic.json
```

The scorer uses the pinned [BEM reproduction](https://huggingface.co/kortukov/answer-equivalence-bem). Agreement requires a score of at least 0.5 in both directions. Check its decisions before treating it as a quality gate. Reports from older runs need `--test-data` when they omit the question text.
