# Offline KV translation

Evaluate a frozen SmolLM2-360M → Qwen2.5-0.5B translator. SmolLM prefills the document, the mapper translates its K/V tensors at causal token boundaries, and Qwen processes the question and answers. Native Qwen split-cache replay checks the harness before each comparison. The experiment is separate from the inference runner.

## Run

Use an account with read access to the private Tandem dataset. Runs pin both base models, the test split, normalization, and the plain and distilled ridge weights.

~~~sh
uvx hf auth login
cd src/experiments/kv_translation
uv run --locked --script run.py run context_only --out ~/tandem-runs
uv run --locked --script run.py judge ~/tandem-runs/context_only-<time>
~~~

Runs default to 120 cases on CUDA in float32, which reproduces the archived answers. Use `--device cpu --count 1` for a smoke test. In bfloat16, native split-cache replay can diverge from full prefill, so use it only for timing. Each run writes `config.json` (commit, versions, pins), `rows.jsonl`, and `summary.json`. Every case first checks that native split-cache replay matches native full prefill.

## Layout

| File | Role |
| --- | --- |
| `harness.py` | Pinned models and data; Qwen forward, greedy generation, teacher-forced loss, replay check |
| `mapper.py` | Token alignment, K/V capture, the affine map, and cache construction |
| `data.py` | `Case` and the pinned HotpotQA test cases |
| `conditions.py` | Ways to build the prefix cache: question only, translated, and controls |
| `metrics.py` | Consistency with native Qwen, optional gold F1, and bootstrap summaries |
| `judge.py` | Qwen2.5-32B judge for semantic agreement with native Qwen |
| `experiments/` | One file per experiment: its cases and conditions |

To add an experiment, add `experiments/<name>.py` with `cases()` and `CONDITIONS`. To add a condition, add a function to `conditions.py`.

## Metrics

Native Qwen is the reference. Per condition, a run reports greedy exact match, first-token top-1 agreement and KL, the extra loss on native's answer (`nll_gap`), and whether the first-sentence entities, numbers, and yes/no match native's. The judge adds `agree`, which requires a SAME verdict and matching entities. Gold F1 is a secondary column.

`run.py bem answers.json --method distilled` rescores the [archived diagnostic answers](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/49ce7e032245e39fedad8d54059f8520735a875d/experiments/context-diagnostic-20261113) with BEM, which produced the findings below. BEM can accept answers that name the wrong entities, so new runs use the judge.

## Findings

- **Context-only handoff.** On 120 prompts, Distilled-8000 reached 62.5% agreement with native Qwen and 47.5% gold correctness. Plain ridge reached 54.2% and 39.2%. Gold correctness was 64.2% for native Qwen, 35.0% with the question alone, and 28.3% with a shuffled cache.
- **Fact transfer.** With the question fixed and one digit changed in the source document, the distilled map followed the change on 120/120 pairs. Plain ridge followed it on 1/120. This was a two-candidate probe.
- **Reconstruction versus behavior.** Distillation increased key and value reconstruction error across all 24 target layers on a 12-prompt subset, while lowering continuation KL from 2.27 to 1.29.
- **Two-hop QA.** On 89 questions where BM25 missed one supporting passage, retrieval alone scored 44.9% against gold. Adding the translated cache reached 47.2% (paired 95% CI for the gain: −6.7 to +11.2 percentage points) and lowered token F1. Supplying the missing text reached 64.0%. The context-only multi-question training trial scored 40.4%, so we skipped the integrated runner benchmark.
- **Latency.** An offline Grace H100 run measured 149 ms to first token with the distilled cache versus 820 ms with native prefill at about 7.7K Qwen tokens. The source had already processed the question and its cache was GPU-resident. This timing excludes source prefill and routing.

[Pinned training splits, weights, and scale results](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm/tree/27744a02d7d5f0abe61216612c46ec3b2a0bb52e/experiments/ridge-32k.C70bPO) · [ML report and split audit](https://github.com/tandemkv/Tandem/pull/91)
