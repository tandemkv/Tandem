"""Cache-mapper contract, independent of any inference framework."""
from typing import Any, Protocol

from src.models.interfaces import AnswerModel


class UnsupportedTranslationError(ValueError):
    """The supplied mapper cannot translate a particular source/target pair."""

    def __init__(self, source_model: str, target_model: str):
        self.source_model = source_model
        self.target_model = target_model
        super().__init__(f"cache translation from {source_model!r} to {target_model!r} is unsupported")


class KVCacheMapper(Protocol):
    def __call__(self, *, cache: Any, source_model: str, target_model: AnswerModel,
                 conversation_text: str, cached_token_ids: list[int],
                 prefix_boundary: int) -> tuple[Any, list[int]]:
        """Return a target cache and nonempty, exact uncached target token IDs.

        Preserve the source cache. source_model is its checkpoint/provenance
        name. prefix_boundary is a character offset into conversation_text.
        The mapper owns alignment, framing and target cache provenance; it must
        raise UnsupportedTranslationError(source_model, target_model.model_name)
        before modifying any state if the direction is unsupported. Other
        failures propagate without fallback. For ConversationManager, tag
        the translated prefix cache with exact tandem_token_ids as well.
        """
        ...
