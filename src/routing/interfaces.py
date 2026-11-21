"""Classifier contracts; implementations need not inherit from these types."""
from typing import Protocol, TypedDict


class ClassificationResult(TypedDict, total=False):
    """Supply task for task routing, complexity for complexity routing, or both.

    Task labels are defined by the caller's task_models mapping. Complexity is
    a finite score on the classifier's own scale; larger means more complex.
    Confidence and dimensions are optional diagnostics, unused by routing.
    """

    task: str
    complexity: float
    task_confidence: float
    dimensions: dict[str, float]


class Classifier(Protocol):
    def __call__(self, prompt: str) -> ClassificationResult:
        """Classify one nonempty user prompt."""
        ...
