"""Answer-model contract, independent of any inference framework."""
from typing import Any, Literal, Protocol


class AnswerModel(Protocol):
    """Generate text while preserving input caches and exposing the result cache.

    model_name is a stable, unique provenance identifier. Caches used for handoff
    expose get_seq_length(); generated caches carry tandem_model_name. For
    ConversationManager, also retain exact tandem_token_ids and the boolean
    tandem_turn_complete. Concrete cache storage belongs to the implementation.
    """

    model_name: str

    def run(self, prompt: str, *, kv_cache: Any = None, is_suffix: bool = False,
            max_new_tokens: int = 100, prompt_mode: Literal["chat", "plain"] = "chat") -> str: ...

    def run_tokens(self, tokens: list[int], *, kv_cache: Any = None,
                   max_new_tokens: int = 100) -> str: ...

    def get_kv_cache(self) -> Any: ...
