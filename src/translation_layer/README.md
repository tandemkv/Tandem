# Local inference runner

Run from the repository root using the experiment environment (`torch` and
`transformers`; the local regression suite was checked with Transformers 5.20.1).
Models and tokenizers load lazily and remain in memory per runner. The first use
may download weights. Calls return `(answer, cache)`; history is reused only
when explicitly supplied. Generation is greedy, batch size one. CPU float32 is the default; configure
`device`, `dtype` (a torch dtype), and `model_revisions` (alias to revision) on
the runner for cluster/training parity. Each revision applies to both weights
and tokenizer. `Model` accepts `device`, `dtype`, and `revision` directly.

```python
from src.translation_layer.inference_runner import InferenceRunner, MODEL_NAMES

runner = InferenceRunner()
prompt = "Say hello."
current_model = "qwen"
answer, cache = runner.receive_prompt(prompt, conversation_text=prompt, current_model=current_model)
print(answer)
current_model = next(alias for alias, name in MODEL_NAMES.items()
                     if name == cache.tandem_model_name)

# Both supported Instruct checkpoints use ChatML. Include the new turn's framing
# but no second system prompt. A completed answer includes its final EOS token.
suffix = (
    "\n<|im_start|>user\nWhat did I ask you to say?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
followup, next_cache = runner.receive_prompt(
    suffix, conversation_text=prompt + answer + suffix,
    current_model=current_model, kv_cache=cache, is_suffix=True,
)
print(followup)
print(cache.get_seq_length(), next_cache.get_seq_length())
print(next_cache.layers[0].keys.shape, next_cache.layers[0].values.shape)
```

If generation reaches `max_new_tokens` before EOS, the cache is an unfinished
assistant turn. Continue that turn with an appropriate suffix or explicitly add
its `<|im_end|>` before the next user turn. For exact split-prefix replay, use
`Model.run_tokens(uncached_token_ids, kv_cache=cache)`; concatenate token IDs
rather than decoding and retokenizing the boundary.

Incoming caches are copied before inference. Both EOS and the last token at the
generation limit are fed through the model, so the returned cache includes the
entire answer. `Model.generated_tokens` exposes the exact generated IDs;
`Model.get_kv_cache()` exposes the resulting DynamicCache and its tensors.
The constructor's optional cache is used once; explicit per-call caches can be
supplied on every request. The default `prompt_mode="chat"` uses
`apply_chat_template`. Set `InferenceRunner(prompt_mode="plain")` or
`Model.run(..., prompt_mode="plain")` to encode raw training text with
`add_special_tokens=False`, without a chat template. Cached text calls in
either mode require `is_suffix=True`.

Caches produced here carry a checkpoint name. For external untagged caches,
the required `current_model="qwen"` or `"smollm"` identifies the source; the caller must ensure
those tensors represent that checkpoint's history and tokenizer. Shape, dtype,
device, finite values, context length, and tagged provenance are validated.
Unknown choices and incompatible caches raise `ValueError`. A failed call does
not mutate the input history.

`current_model` is required on every call and identifies the session model before
routing. The router always selects the target; a different selection triggers
translation when a cache is supplied. After a successful call, use the returned
cache's checkpoint tag to update the session's current model.

Cross-model cache reuse requires an injected `cache_adapter` callable (#73).
The runner loads the target before calling `_translate_cache`; the adapter
returns `(translated_cache, uncached_target_token_ids)`. These IDs go directly
to `run_tokens`, without retokenization or another chat template.

```python
runner = InferenceRunner(
    prompt_mode="plain", cache_adapter=adapter,
    device="cuda:0", dtype=torch.bfloat16,
    model_revisions={"qwen": "<commit>", "smollm": "<commit>"},
)
answer, target_cache = runner.receive_prompt(
    full_text, current_model="smollm", kv_cache=source_cache,
    conversation_text=full_text, cached_token_ids=source_ids,
    prefix_boundary=len(prefix_text),
)
```

Import `torch` for this example. `conversation_text` is required on every call,
even when no cache is supplied or the model stays the same, because routing is
decided inside the runner. It is the complete source text, including the new
prompt (the uncached continuation). On the first call, pass `conversation_text=prompt`. `prefix_boundary` is the cached prefix's
end as a Python character offset into that text (not a byte or token offset).
`cached_token_ids` contains every exact source ID represented by the cache,
including framing and generated tokens; its length must match the cache.
Callers must retain this metadata; decoding tokens cannot reliably recover the
original text. Routing still uses the `prompt` argument.

The callable receives keyword arguments `cache`, `source_model` (checkpoint
name), `target_model` (loaded Model wrapper), `conversation_text`,
`cached_token_ids`, `prefix_boundary`, and `prompt_mode`. The wrapper exposes
its tokenizer, revision, and underlying model's device/dtype. The adapter
owns checkpoint selection, tokenizer alignment, chat framing where applicable,
key-position conversion, and target cache provenance; it must not mutate the
source cache. Return a nonempty list of uncached target IDs and a compatible
DynamicCache. Same-model calls bypass the adapter.

Without an adapter, cross-model reuse raises an explicit error. The learned
adapter remains #73's scope; Qwen → SmolLM requires its own checkpoint (#74).
Switching models without a cache already works. The runner is intended for
sequential local calls, not concurrent requests.

```sh
src/venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
TANDEM_REAL_MODEL_TEST=1 src/venv/bin/python -m unittest discover -s test -p test_model.py -v
```

The default model tests use a tiny randomly initialized Qwen architecture with
no downloads. The opt-in test uses `Qwen/Qwen2.5-0.5B-Instruct` and compares cached
continuation tokens against a full-history run while checking the input tensors
remain unchanged.

Offline tests also force both switch directions through a native-prefill test
adapter and compare against a full native target run using identical target
IDs. This verifies handoff mechanics, not learned translation quality. Real
checkpoint switch evaluation remains part of #75.
