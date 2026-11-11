"""Lazy model selection and explicit cache handoff for the local PoC."""
from collections.abc import Mapping
from typing import Literal

import torch

from src.models.model import Model
from src.routing import router

MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


class InferenceRunner:
    
    def __init__(
        self,
        task_models: Mapping[str, str] | None = None,
        *,
        mode: Literal["task", "complexity"] = "task",
        complexity_threshold: float = 0.5,
        prompt_mode: Literal["chat", "plain"] = "chat",
        device="cpu",
        dtype=torch.float32,
        model_revisions: Mapping[str, str] | None = None,
        cache_adapter=None,
    ) -> None:
        if prompt_mode not in ("chat", "plain"):
            raise ValueError("prompt_mode must be chat or plain")
        self.prompt_mode = prompt_mode
        self.device = device
        self.dtype = dtype
        self.model_revisions = dict(model_revisions or {})
        for alias in self.model_revisions:
            self._validate_model(alias)
        self.cache_adapter = cache_adapter
        self.task_models = task_models
        self.mode = mode
        self.complexity_threshold = complexity_threshold
        self._models = {}


    @staticmethod
    def _validate_model(choice):
        if choice not in MODEL_NAMES:
            raise ValueError(f"unknown model {choice!r}; choose from {', '.join(MODEL_NAMES)}")


    def receive_prompt(self, prompt: str, *, conversation_text: str, current_model: str, kv_cache=None,
                       is_suffix=False, max_new_tokens=100,
                       cached_token_ids=None, prefix_boundary=None) -> tuple[str, object]:
        """Return (answer, cache). No conversation state is reused implicitly.

        current_model identifies the session's current model and supplied cache's
        source alias. The router selects the model for every request. Tagged
        caches must agree with current_model. conversation_text is required on every
        call and contains the complete conversation text, including the new,
        unprocessed prompt,
        so callers need not predict routing. For translation, prefix_boundary is
        its cached prefix end (a Python character offset), and cached_token_ids are the exact
        source IDs represented by the cache, including any framing/generated IDs.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if not isinstance(conversation_text, str) or not conversation_text.strip():
            raise ValueError("conversation_text must be nonempty text including the new prompt")
        self._validate_model(current_model)
        choice = router.route(
            prompt, mode=self.mode, threshold=self.complexity_threshold,
            task_models=self.task_models,
        )
        self._validate_model(choice)
        if (kv_cache is not None
                and getattr(kv_cache, "tandem_model_name", None) not in
                (None, MODEL_NAMES[current_model])):
            raise ValueError("current_model disagrees with cache provenance")
        switching = choice != current_model
        translating = switching and kv_cache is not None
        if translating and self.cache_adapter is not None:
            if (type(prefix_boundary) is not int
                    or not 0 <= prefix_boundary <= len(conversation_text)):
                raise ValueError("translation requires conversation_text and a character prefix_boundary")
            if (cached_token_ids is None or len(cached_token_ids) != kv_cache.get_seq_length()
                    or any(type(token) is not int or token < 0 for token in cached_token_ids)):
                raise ValueError("cached_token_ids must match the source cache length")
        if choice not in self._models:
            self._models[choice] = Model(
                model_name=MODEL_NAMES[choice], device=self.device, dtype=self.dtype,
                revision=self.model_revisions.get(choice),
            )
        model = self._models[choice]
        if translating:
            translated_cache, target_tokens = self._translate_cache(
                kv_cache, MODEL_NAMES[current_model], model, conversation_text=conversation_text,
                cached_token_ids=cached_token_ids, prefix_boundary=prefix_boundary,
            )
            answer = model.run_tokens(target_tokens, kv_cache=translated_cache,
                                      max_new_tokens=max_new_tokens)
        else:
            answer = model.run(prompt, kv_cache=kv_cache, is_suffix=is_suffix,
                               max_new_tokens=max_new_tokens, prompt_mode=self.prompt_mode)
        return answer, model.get_kv_cache()


    def _translate_cache(self, cache, source_model, target_model, *, conversation_text,
                         cached_token_ids, prefix_boundary):
        """Adapter returns (target DynamicCache, exact uncached target IDs).

        target_model is the loaded Model wrapper (tokenizer, device, dtype and
        revision available). The adapter owns alignment, checkpoint selection,
        key-position conversion and target cache provenance. It must preserve
        the input cache. conversation_text includes the uncached continuation; the
        adapter formats chat mode exactly once when preparing target tokens.
        """
        if self.cache_adapter is None:
            raise ValueError(
                f"cache from {source_model} is incompatible with {target_model.model_name}; "
                "cross-model cache translation is not available yet (#73)"
            )
        return self.cache_adapter(
            cache=cache, source_model=source_model, target_model=target_model,
            conversation_text=conversation_text, cached_token_ids=cached_token_ids,
            prefix_boundary=prefix_boundary, prompt_mode=self.prompt_mode,
        )
