# Optional model adapters

[`nvidia_classifier.py`](nvidia_classifier.py) implements the
[`Classifier` protocol](../routing/interfaces.py) using the
[NVIDIA Prompt Task and Complexity Classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier).
Import it only when choosing this implementation:

```python
from src.adapters.nvidia_classifier import PromptClassifier

classifier = PromptClassifier()
prediction = classifier("Explain binary search")
# task, task_confidence, complexity, dimensions
```

This adapter needs `torch`, `transformers`, `huggingface-hub`, and `safetensors`.
Construction downloads the classifier weights (roughly 740 MB), tokenizer, and
backbone architecture configuration, then loads the model on CPU. It does not
download base DeBERTa weights separately. Reuse the instance across calls;
later constructions use the Hugging Face disk cache. The core router and runner
do not import this adapter or require these dependencies.

The classifier is intended for English and considers the first 512 tokenizer
tokens. Pass the current request, not the full conversation. Complexity uses
NVIDIA's weighted sum of creativity (0.35), reasoning (0.25), constraints (0.15),
domain knowledge (0.15), contextual knowledge (0.05), and few-shot count (0.05),
with the published normalization and rounding. Few-shot count ranges up to 5,
so the maximum score is **1.2**, not 1.0. This score is not a probability of
answer-model failure; calibrate the routing threshold on representative prompts.

Weights are under the NVIDIA Open Model License. The implementation follows the
architecture/scoring in the official model card, uses strict safetensors loading,
and pins classifier revision `fea1121511eafabaf7dd6fc66863dcb04f74defb` and backbone
configuration revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.

The optional Hugging Face answer-model implementation remains in
[`src/models/model.py`](../models/model.py). It implements
[`AnswerModel`](../models/interfaces.py) using PyTorch and Transformers.
The [demo setup](../experiments/poc/demo_setup.py) constructs this implementation
with Qwen and SmolLM and injects the NVIDIA classifier. Custom implementations
only need to satisfy the core protocols; they do not need to subclass the
provided implementations.
