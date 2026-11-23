"""Route between caller-supplied models with explicit cache handoff."""
from collections.abc import Mapping
from typing import Literal

from src.models.interfaces import AnswerModel
from src.routing import router
from src.routing.interfaces import Classifier
from src.translation_layer.interfaces import KVCacheMapper, UnsupportedTranslationError


class InferenceRunner:
    """Use model_a for complex prompts/fallback and model_b for simpler ones.

    Answer models implement AnswerModel. classifier_model(prompt) returns
    task/complexity predictions according to the Classifier protocol.
    kv_cache_mapper follows the keyword contract documented in _translate_cache;
    pass None to disable cross-model cache reuse (switches then run natively).
    Prompts are encoded verbatim, so callers include any chat framing.
    Loading belongs to the caller.
    """

    def __init__(
        self,
        model_a: AnswerModel,
        model_b: AnswerModel,
        classifier_model: Classifier,
        kv_cache_mapper: KVCacheMapper | None,
        *,
        task_models: Mapping[str, str] | None = None,
        mode: Literal["task", "complexity"] = "task",
        complexity_threshold: float = 0.5,
    ) -> None:
        self._models = {"model_a": model_a, "model_b": model_b}
        self.model_names = {}
        for alias, model in self._models.items():
            name = getattr(model, "model_name", None)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{alias} must have a nonempty model_name")
            for method in ("run", "run_tokens", "get_kv_cache"):
                if not callable(getattr(model, method, None)):
                    raise TypeError(f"{alias} must provide callable {method}()")
            self.model_names[alias] = name
        if model_a.model_name == model_b.model_name:
            raise ValueError("answer models must have distinct model_name values for cache provenance")
        if not callable(classifier_model):
            raise TypeError("classifier_model must be callable")
        if kv_cache_mapper is not None and not callable(kv_cache_mapper):
            raise TypeError("kv_cache_mapper must be callable or None")
        self.classifier_model = classifier_model
        self.kv_cache_mapper = kv_cache_mapper
        router.validate_configuration(mode=mode, threshold=complexity_threshold,
                                      task_models=task_models)
        self.task_models = dict(task_models) if task_models is not None else None
        self.mode = mode
        self.complexity_threshold = complexity_threshold


    def _validate_model(self, choice):
        if not isinstance(choice, str) or choice not in self._models:
            raise ValueError(f"unknown model {choice!r}; choose from {', '.join(self._models)}")


    def receive_prompt(self, prompt: str, *, conversation_text: str, current_model: str, kv_cache=None,
                       is_suffix=False, max_new_tokens=100,
                       cached_token_ids=None, prefix_boundary=None,
                       uncached_suffix: str | None = None) -> tuple[str, object]:
        """Return (answer, cache). No conversation state is reused implicitly.

        current_model identifies the session's current model and supplied cache's
        source alias. The router selects the model for every request. Tagged
        caches must agree with current_model. conversation_text is required on every
        call and contains the complete conversation text, including the new,
        unprocessed prompt,
        so callers need not predict routing. For translation, prefix_boundary is
        its cached prefix end (a Python character offset), and cached_token_ids are the exact
        source IDs represented by the cache, including any framing/generated IDs.
        uncached_suffix allows routing on the user message while executing
        separately framed text. With a cache it requires is_suffix=True.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if not isinstance(conversation_text, str) or not conversation_text.strip():
            raise ValueError("conversation_text must be nonempty text including the new prompt")
        if uncached_suffix is not None and (
                (kv_cache is not None and not is_suffix)
                or not isinstance(uncached_suffix, str) or not uncached_suffix.strip()):
            raise ValueError("uncached_suffix requires nonempty text, and is_suffix=True with a cache")
        self._validate_model(current_model)
        choice = router.route(
            prompt, mode=self.mode, threshold=self.complexity_threshold,
            task_models=self.task_models,
            classifier_model=self.classifier_model, model_a="model_a", model_b="model_b",
        )
        self._validate_model(choice)
        if (kv_cache is not None
                and getattr(kv_cache, "tandem_model_name", None) not in
                (None, self.model_names[current_model])):
            raise ValueError("current_model disagrees with cache provenance")
        switching = choice != current_model
        translating = switching and kv_cache is not None
        if translating and self.kv_cache_mapper is not None:
            if (type(prefix_boundary) is not int
                    or not 0 <= prefix_boundary <= len(conversation_text)):
                raise ValueError("translation requires conversation_text and a character prefix_boundary")
            if (cached_token_ids is None or len(cached_token_ids) != kv_cache.get_seq_length()
                    or any(type(token) is not int or token < 0 for token in cached_token_ids)):
                raise ValueError("cached_token_ids must match the source cache length")
        model = self._models[choice]
        translated_cache = None
        if translating:
            translated_cache, target_tokens = self._translate_cache(
                kv_cache, self.model_names[current_model], model, conversation_text=conversation_text,
                cached_token_ids=cached_token_ids, prefix_boundary=prefix_boundary,
            )
        if translated_cache is not None:
            answer = model.run_tokens(target_tokens, kv_cache=translated_cache,
                                      max_new_tokens=max_new_tokens)
        elif translating:
            # No map for this direction: process the whole conversation natively (FR-06).
            answer = model.run(conversation_text, max_new_tokens=max_new_tokens)
        else:
            answer = model.run(prompt if uncached_suffix is None else uncached_suffix,
                               kv_cache=kv_cache, is_suffix=is_suffix,
                               max_new_tokens=max_new_tokens)
        return answer, model.get_kv_cache()


    def _translate_cache(self, cache, source_model, target_model, *, conversation_text,
                         cached_token_ids, prefix_boundary):
        """Mapper returns (target cache, exact uncached target IDs).

        target_model is the supplied AnswerModel. The mapper must support this
        model pair and owns alignment, checkpoint selection,
        key-position conversion and target cache provenance. It must preserve
        the input cache. conversation_text includes the uncached continuation.
        A missing mapper or unsupported direction returns (None, None) so the
        caller processes the conversation natively.
        """
        if self.kv_cache_mapper is None:
            return None, None
        try:
            return self.kv_cache_mapper(
                cache=cache, source_model=source_model, target_model=target_model,
                conversation_text=conversation_text, cached_token_ids=cached_token_ids,
                prefix_boundary=prefix_boundary,
            )
        except UnsupportedTranslationError:
            return None, None
