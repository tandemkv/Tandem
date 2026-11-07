"""Lazy model selection and explicit cache handoff for the local PoC."""
from collections.abc import Mapping
from typing import Literal

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
    ) -> None:
        self._validate_model(starting_model)
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
                       cache_model=None, is_suffix=False, max_new_tokens=100) -> tuple[str, object]:
        """Return (answer, cache). No conversation state is reused implicitly.

        model_choice can be a router result; otherwise route this prompt here.
        cache_model identifies an external cache's source alias. Caches produced
        here also carry their checkpoint name, so cross-model reuse is rejected.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        choice = model_choice if model_choice is not None else router.route(
            prompt, mode=self.mode, threshold=self.complexity_threshold,
            task_models=self.task_models,
        )
        self._validate_model(choice)
        if kv_cache is not None:
            owner = getattr(kv_cache, "tandem_model_name", None)
            if cache_model is not None:
                self._validate_model(cache_model)
                if owner is not None and owner != MODEL_NAMES[cache_model]:
                    raise ValueError("cache_model disagrees with cache provenance")
                owner = MODEL_NAMES[cache_model]
            if owner is None:
                raise ValueError("cache_model is required for an untagged external cache")
            if owner != MODEL_NAMES[choice]:
                kv_cache = self._translate_cache(kv_cache, owner, MODEL_NAMES[choice])
        if choice not in self._models:
            self._models[choice] = Model(model_name=MODEL_NAMES[choice])
        model = self._models[choice]
        answer = model.run(prompt, kv_cache=kv_cache, is_suffix=is_suffix,
                           max_new_tokens=max_new_tokens)
        self.current_model = choice
        return answer, model.get_kv_cache()

    def _translate_cache(self, cache, source_model, target_model):
        # Integrate the translator and its to_cache() when #56 is available.
        raise ValueError(
            f"cache from {source_model} is incompatible with {target_model}; "
            "cross-model cache translation is not available yet (#56)"
        )
