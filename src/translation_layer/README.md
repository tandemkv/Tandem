# Local inference runner

For stateful conversations, use `ConversationManager`:

```python
from src.experiments.poc.demo_setup import build_demo_runner
from src.translation_layer.conversation_manager import ConversationManager

# Optional Qwen/SmolLM/NVIDIA setup with the SmolLM → Qwen cache mapper.
runner = build_demo_runner()
conversation = ConversationManager(runner, current_model="model_a")
answer = conversation.send("Explain gravity.")
answer = conversation.send("How does that affect time?", max_new_tokens=100)
```

The runner requires four supplied dependencies: two answer models, a classifier,
and a KV cache mapper (or explicit `None` to disable translation). The demo is
optional; for your own implementations, construct the runner directly:

```python
from src.translation_layer.inference_runner import InferenceRunner

runner = InferenceRunner(
    model_a=answer_model_a, model_b=answer_model_b,
    classifier_model=classifier, kv_cache_mapper=mapper,
    task_models={"your label": "model_a", "another label": "model_b"},
)
```

Implement these protocols without inheriting from any Tandem class:

- [`AnswerModel`](../models/interfaces.py): `model_name`, `run`, `run_tokens`,
  and `get_kv_cache`. Names must be stable, nonempty, and distinct for provenance.
- [`Classifier`](../routing/interfaces.py): calling it with a prompt returns a
  `ClassificationResult` with `task` and/or `complexity` for the chosen mode.
- [`KVCacheMapper`](interfaces.py): translates a cache and returns exact uncached
  target token IDs, following the handoff contract below.

Supplied instances are reused; loading belongs to the caller. The core imports
only standard-library dependencies. The optional NVIDIA adapter and Hugging Face
answer model are imported only by users who select them, such as the demo.

Routing aliases are `model_a` and `model_b`, independent of checkpoint names.
Task mode requires an explicit `task_models` mapping. Complexity mode selects
`model_a` at or above `complexity_threshold`, otherwise `model_b`; the threshold
uses the classifier's score scale. Classifier inference failures fall back to
`model_a`. The demo factory supplies the NVIDIA label mapping.

Initialization validates answer-model names and required callable methods,
classifier/mapper callability, routing mode, task mappings, and the complexity
threshold when complexity mode is selected. Invalid settings fail before any
inference. Protocols describe the method signatures and cache semantics that
implementations must honor; they do not require inheritance.

The manager owns `history` (user/assistant messages), `kv_cache`,
`current_model`, `cached_token_ids`, `conversation_text`, and `prefix_boundary`.
Each successful send updates these together; failed sends preserve prior state.
Routing uses the new user message. Each turn is written as ChatML text
(`<|im_start|>user ... <|im_start|>assistant`), so answer models must be
ChatML-compatible. A completed answer's `<|im_end|>` stays in
`conversation_text`, so the text matches the cached token IDs; an unfinished
turn is closed when generation hits the token limit. Managers can share a
runner for sequential calls while retaining independent histories. Create a new
manager to start a new conversation.

Routing may select a different model from `current_model`. A switch is
translated when the mapper supports the direction; otherwise the full
conversation runs natively on the new model. Adapters
used with the manager must attach `tandem_token_ids` to the translated prefix
cache: the exact target token IDs represented by that cache. `Model.run_tokens`
appends the uncached and generated IDs to this metadata. Without exact prefix
IDs, the manager rejects the result and retains its previous state.

Run from the repository root using the experiment environment (`torch` and
`transformers`; the local regression suite was checked with Transformers 5.20.1).
Constructing the supplied models may download weights. Calls return
`(answer, cache)`; history is reused only when explicitly supplied. The provided
`Model` uses greedy generation, batch size one, and defaults to CPU float32.
Configure `device`, `dtype` (a torch dtype), and `revision` on each
`Model` for cluster/training parity. Revisions apply to both weights and tokenizer.

```python
# Reuse the runner constructed above.
# Text is encoded verbatim, so write the ChatML framing that Instruct models expect.
message = "Say hello."
turn = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
current_model = "model_a"
answer, cache = runner.receive_prompt(message, conversation_text=turn, uncached_suffix=turn,
                                      current_model=current_model)
print(answer)
current_model = next(alias for alias, name in runner.model_names.items()
                     if name == cache.tandem_model_name)

# A completed answer's cache includes its final <|im_end|>, so the text does too.
suffix = (
    "\n<|im_start|>user\nWhat did I ask you to say?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
followup, next_cache = runner.receive_prompt(
    suffix, conversation_text=turn + answer + "<|im_end|>" + suffix,
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
supplied on every request. `Model.run` encodes text verbatim with
`add_special_tokens=False`, so include any chat framing in the text. Cached
text calls require `is_suffix=True`.

Caches produced here carry a checkpoint name. For external untagged caches,
the required `current_model="model_a"` or `"model_b"` identifies the source; the caller must ensure
those tensors represent that checkpoint's history and tokenizer. Shape, dtype,
device, finite values, context length, and tagged provenance are validated.
Unknown choices and incompatible caches raise `ValueError`. A failed call does
not mutate the input history.

`current_model` is required on every call and identifies the session model before
routing. The router always selects the target; a different selection triggers
translation when a cache is supplied. After a successful call, use the returned
cache's checkpoint tag to update the session's current model.

Cross-model cache reuse requires an injected `kv_cache_mapper` callable (#73).
Without one, switches run natively. The runner passes the supplied target model to `_translate_cache`; the mapper
returns `(translated_cache, uncached_target_token_ids)`. These IDs go directly
to `run_tokens`, without retokenization.

```python
runner = InferenceRunner(
    model_a=model_a, model_b=model_b, classifier_model=classifier,
    kv_cache_mapper=mapper, mode="complexity",
)
answer, target_cache = runner.receive_prompt(
    full_text, current_model="model_b", kv_cache=source_cache,
    conversation_text=full_text, cached_token_ids=source_ids,
    prefix_boundary=len(prefix_text),
)
```

Here `model_a`, `model_b`, `classifier`, and `mapper` are your supplied instances.
`conversation_text` is required on every call,
even when no cache is supplied or the model stays the same, because routing is
decided inside the runner. It is the complete source text, including the new
prompt (the uncached continuation). On the first call, pass `conversation_text=prompt`. `prefix_boundary` is the cached prefix's
end as a Python character offset into that text (not a byte or token offset).
`cached_token_ids` contains every exact source ID represented by the cache,
including framing and generated tokens; its length must match the cache.
Callers must retain this metadata; decoding tokens cannot reliably recover the
original text. Routing still uses the `prompt` argument.

The mapper is a callable model or wrapper around a learned mapper. It receives
keyword arguments `cache`, `source_model` (checkpoint
name), `target_model` (supplied `AnswerModel`), `conversation_text`,
`cached_token_ids`, and `prefix_boundary`. The mapper must support
the supplied answer-model pair. The optional `Model` implementation exposes its
tokenizer, revision, and underlying model's device/dtype. The adapter
owns checkpoint selection, tokenizer alignment,
key-position conversion, and target cache provenance; it must not mutate the
source cache. Return a nonempty list of uncached target IDs and a compatible
DynamicCache. Same-model calls bypass the adapter.

For an unsupported direction, mappers must raise
`UnsupportedTranslationError(source_model, target_model.model_name)` from
`src.translation_layer.interfaces` before modifying the source cache or any
other state. The runner catches it and runs the full `conversation_text` on
the target without a cache (native prefill, FR-06). Other mapper failures
propagate, and the conversation manager keeps its prior history and cache so
the caller can retry.

`CacheMapper` holds one pinned affine map per direction; today SmolLM → Qwen
uses the Distilled-8000 checkpoint from the offline evaluation. Its constructor verifies the downloaded files and
requires read access to the private Hugging Face dataset
`tandemkv/paired-kv-qwen-smollm`. Construct it once and reuse it with the same models.

```python
from src.translation_layer.cache_mapper import CacheMapper, SMOLLM, QWEN
from src.models.model import Model
from src.translation_layer.inference_runner import InferenceRunner

smol = Model(model_name=SMOLLM[0], revision=SMOLLM[1])
qwen = Model(model_name=QWEN[0], revision=QWEN[1])
runner = InferenceRunner(qwen, smol, classifier_model=lambda _: {"task": "question"},
                         task_models={"question": "model_a"},
                         kv_cache_mapper=CacheMapper([smol, qwen]))
```

Supply a revision-tagged SmolLM cache and its exact text prefix using
the handoff arguments above. Qwen → SmolLM needs its own checkpoint, added as a `DIRECTIONS` entry (#74).
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
