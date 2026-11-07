# Local inference runner

Run from the repository root using the experiment environment (`torch` and
`transformers`; the local regression suite was checked with Transformers 5.20.1).
Models and tokenizers load lazily and remain in memory per runner. The first use
may download weights. Calls return `(answer, cache)`; history is reused only
when explicitly supplied. Generation is greedy, batch size one, on CPU.

```python
from src.routing import router
from src.translation_layer.inference_runner import InferenceRunner

runner = InferenceRunner()
prompt = "Say hello."
choice = router.route(prompt)
answer, cache = runner.receive_prompt(prompt, model_choice=choice)
print(answer)

# Both supported Instruct checkpoints use ChatML. Include the new turn's framing
# but no second system prompt. A completed answer includes its final EOS token.
suffix = (
    "\n<|im_start|>user\nWhat did I ask you to say?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
followup, next_cache = runner.receive_prompt(
    suffix, model_choice=choice, kv_cache=cache, is_suffix=True,
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
supplied on every request. Plain prompts use `apply_chat_template`.

Caches produced here carry a checkpoint name. For external untagged caches,
the runner requires `cache_model="qwen"` or `"smollm"`; the caller must ensure
those tensors represent that checkpoint's history and tokenizer. Shape, dtype,
device, finite values, context length, and tagged provenance are validated.
Unknown choices and incompatible caches raise `ValueError`. A failed call does
not change the runner's current selection or mutate the input history.

Cross-model cache reuse raises an explicit error. `_translate_cache` is the
integration point for #56's translator and `to_cache()`; translation is not
implemented here. Switching models without a cache already works. The runner
is intended for sequential local calls, not concurrent requests.

```sh
src/venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
TANDEM_REAL_MODEL_TEST=1 src/venv/bin/python -m unittest discover -s test -p test_model.py -v
```

The default model tests use a tiny randomly initialized Qwen architecture with
no downloads. The opt-in test uses `Qwen/Qwen2.5-0.5B-Instruct` and compares cached
continuation tokens against a full-history run while checking the input tensors
remain unchanged.
