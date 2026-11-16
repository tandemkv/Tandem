"""Named, directional checkpoints using the existing cache-mapper contract."""
from dataclasses import dataclass

from src.translation_layer.interfaces import KVCacheMapper


@dataclass(frozen=True)
class TranslationCheckpoint:
    """A loaded mapper and its checkpoint identity for a registered direction.

    The caller must register the direction declared by the checkpoint. The
    adapter still validates revisions, tokenizers and layout before translation.
    """

    checkpoint: str
    mapper: KVCacheMapper

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("checkpoint must be a nonempty identifier")
        if not callable(self.mapper):
            raise TypeError("checkpoint mapper must be callable")


@dataclass(frozen=True)
class TranslationMetadata:
    translation_model: str
    checkpoint: str | None
    source_model: str
    target_model: str
    translated: bool


class InferenceResult(tuple):
    """Backward-compatible (answer, cache) with per-call translation metadata."""

    def __new__(cls, answer, cache, *, metadata: TranslationMetadata):
        result = super().__new__(cls, (answer, cache))
        result.metadata = metadata
        return result
