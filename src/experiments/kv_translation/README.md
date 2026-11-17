# Offline KV translation

Evaluate a frozen SmolLM2-360M → Qwen2.5-0.5B translator. SmolLM prefills the document, the mapper translates its K/V tensors at causal token boundaries, and Qwen processes the question and answers. Native Qwen split-cache replay checks the harness before each comparison. The experiment is separate from the inference runner.

## Run

Use an account with read access to the private Tandem dataset. The replay pins both base models, the test split, normalization, and the plain and distilled ridge weights.

~~~sh
uvx hf auth login
uv run --locked --script src/experiments/kv_translation/evaluate.py replay --output answers.json
uv run --locked --script src/experiments/kv_translation/evaluate.py score answers.json --method distilled
~~~

Replay defaults to 120 prompts on CUDA. Use `--device cpu --count 1` for a smoke test. The scorer accepts JSON or JSONL, including the [archived diagnostic answers](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/49ce7e032245e39fedad8d54059f8520735a875d/experiments/context-diagnostic-20261113). BEM measures bidirectional agreement with native Qwen and scores each answer against gold. It can miss factual differences, so agreement and correctness are reported separately.

## Findings

- **Context-only handoff.** On 120 prompts, Distilled-8000 reached 62.5% agreement with native Qwen and 47.5% gold correctness. Plain ridge reached 54.2% and 39.2%. Gold correctness was 64.2% for native Qwen, 35.0% with the question alone, and 28.3% with a shuffled cache.
- **Fact transfer.** With the question fixed and one digit changed in the source document, the distilled map followed the change on 120/120 pairs. Plain ridge followed it on 1/120. This was a two-candidate probe.
- **Reconstruction versus behavior.** Distillation increased key and value reconstruction error across all 24 target layers on a 12-prompt subset, while lowering continuation KL from 2.27 to 1.29.
- **Two-hop QA.** On 89 questions where BM25 missed one supporting passage, retrieval alone scored 44.9% against gold. Adding the translated cache reached 47.2% (paired 95% CI for the gain: −6.7 to +11.2 percentage points) and lowered token F1. Supplying the missing text reached 64.0%. The context-only multi-question training trial scored 40.4%, so we skipped the integrated runner benchmark.
- **Latency.** An offline Grace H100 run measured 149 ms to first token with the distilled cache versus 820 ms with native prefill at about 7.7K Qwen tokens. The source had already processed the question and its cache was GPU-resident. This timing excludes source prefill and routing.

[Pinned training splits, weights, and scale results](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/27744a02d7d5f0abe61216612c46ec3b2a0bb52e/experiments/ridge-32k.C70bPO) · [ML report and split audit](https://github.com/tandemkv/Tandem/pull/91)
