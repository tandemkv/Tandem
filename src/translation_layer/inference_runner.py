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
        starting_model: str = "qwen",
        prompt_mode: Literal["chat", "plain"] = "chat",
        device="cpu",
        dtype=torch.float32,
        model_revisions: Mapping[str, str] | None = None,
        cache_adapter=None,
    ) -> None:
        self._validate_model(starting_model)
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
        self.current_model = starting_model
        self._models = {}

    @staticmethod
    def _validate_model(choice):
        if choice not in MODEL_NAMES:
            raise ValueError(f"unknown model {choice!r}; choose from {', '.join(MODEL_NAMES)}")

    def receive_prompt(self, prompt: str, *, model_choice=None, kv_cache=None,
                       cache_model=None, is_suffix=False, max_new_tokens=100,
                       original_text=None, cached_token_ids=None, prefix_boundary=None) -> tuple[str, object]:
        """Return (answer, cache). No conversation state is reused implicitly.

        model_choice can be a router result; otherwise route this prompt here.
        cache_model identifies an external cache's source alias. Caches produced
        here also carry their checkpoint name. For translation, original_text
        contains the complete source text, prefix_boundary is its cached prefix
        end (a Python character offset), and cached_token_ids are the exact
        source IDs represented by the cache, including any framing/generated IDs.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        choice = model_choice if model_choice is not None else router.route(
            prompt, mode=self.mode, threshold=self.complexity_threshold,
            task_models=self.task_models,
        )
        self._validate_model(choice)
        owner = None
        if kv_cache is not None:
            owner = getattr(kv_cache, "tandem_model_name", None)
            if cache_model is not None:
                self._validate_model(cache_model)
                if owner is not None and owner != MODEL_NAMES[cache_model]:
                    raise ValueError("cache_model disagrees with cache provenance")
                owner = MODEL_NAMES[cache_model]
            if owner is None:
                raise ValueError("cache_model is required for an untagged external cache")
        switching = kv_cache is not None and owner != MODEL_NAMES[choice]
        if switching and self.cache_adapter is not None:
            if (not isinstance(original_text, str)
                    or type(prefix_boundary) is not int
                    or not 0 <= prefix_boundary <= len(original_text)):
                raise ValueError("translation requires original_text and a character prefix_boundary")
            if (cached_token_ids is None or len(cached_token_ids) != kv_cache.get_seq_length()
                    or any(type(token) is not int or token < 0 for token in cached_token_ids)):
                raise ValueError("cached_token_ids must match the source cache length")
        if choice not in self._models:
            self._models[choice] = Model(
                model_name=MODEL_NAMES[choice], device=self.device, dtype=self.dtype,
                revision=self.model_revisions.get(choice),
            )
        model = self._models[choice]
        if switching:
            translated_cache, target_tokens = self._translate_cache(
                kv_cache, owner, model, original_text=original_text,
                cached_token_ids=cached_token_ids, prefix_boundary=prefix_boundary,
            )
            answer = model.run_tokens(target_tokens, kv_cache=translated_cache,
                                      max_new_tokens=max_new_tokens)
        else:
            answer = model.run(prompt, kv_cache=kv_cache, is_suffix=is_suffix,
                               max_new_tokens=max_new_tokens, prompt_mode=self.prompt_mode)
        self.current_model = choice
        return answer, model.get_kv_cache()

    def _translate_cache(self, cache, source_model, target_model, *, original_text,
                         cached_token_ids, prefix_boundary):
        """Adapter returns (target DynamicCache, exact uncached target IDs).

        target_model is the loaded Model wrapper (tokenizer, device, dtype and
        revision available). The adapter owns alignment, checkpoint selection,
        key-position conversion and target cache provenance. It must preserve
        the input cache. original_text includes the uncached continuation; the
        adapter formats chat mode exactly once when preparing target tokens.
        """
        if self.cache_adapter is None:
            raise ValueError(
                f"cache from {source_model} is incompatible with {target_model.model_name}; "
                "cross-model cache translation is not available yet (#73)"
            )
        return self.cache_adapter(
            cache=cache, source_model=source_model, target_model=target_model,
            original_text=original_text, cached_token_ids=cached_token_ids,
            prefix_boundary=prefix_boundary, prompt_mode=self.prompt_mode,
        )
