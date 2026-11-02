# Prompt → model choice

```python
from src.routing.router import route

route("Write a Python function", mode="task")
route("Write a Python function", mode="complexity", threshold=0.5)
# Both return only "qwen" or "smollm".
```

Neither target model is loaded or called. `qwen` means
`Qwen/Qwen2.5-0.5B-Instruct`; `smollm` means
`HuggingFaceTB/SmolLM2-360M-Instruct`. Your inference code uses the choice and
owns the KV caches.

## Toggle

- **task** (default): NVIDIA's classifier predicts a task label. Code Generation,
  Open QA, Other, and Unknown go to Qwen. Other labels (including Summarization,
  Rewrite, Extraction, and Chatbot) go to SmolLM. Edit `TASK_MODELS` in `router.py`
  to change the policy.
- **complexity**: Qwen if the overall complexity score is at least `threshold`;
  otherwise SmolLM. The default cutoff is 0.5. The threshold is ignored in task
  mode. Both the mapping and cutoff are experimental, not measured performance
  rankings for these target models.

Both modes use the same local
[NVIDIA Prompt Task and Complexity Classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier).
It replaces the previous MiniLM/example-similarity implementation with actual
trained task labels and complexity predictions. The classifier is loaded once
per process, on CPU; switching modes does not load a second model.

Inspect predictions independently when tuning:

```python
from src.routing.router import analyze

print(analyze("Explain how binary search works."))
# task, task_confidence, complexity, dimensions
```

Complexity uses NVIDIA's weighted sum of creativity (0.35), reasoning (0.25),
constraints (0.15), domain knowledge (0.15), contextual knowledge (0.05), and
few-shot count (0.05), with the published normalization and rounding. Few-shot
count ranges up to 5, so the mathematical maximum is **1.2**, not 1.0. The score
is not the probability that a target model will fail. Calibrate the threshold on
representative prompts and measured performance of both target models.

## Run

Use the existing experiment environment from the repository root:

```sh
src/venv/bin/python -m src.routing.router --mode task "Write a Python function"
src/venv/bin/python -m src.routing.router --mode complexity --threshold 0.5 "Write a Python function"
```

For a fresh environment install `torch`, `transformers`, `huggingface-hub`, and
`safetensors`. The first valid call downloads the classifier weights (roughly
740 MB), tokenizer, and backbone architecture configuration. It does not download
base DeBERTa weights separately. Later runs use the Hugging Face disk cache;
within a process the model stays in memory. No servers, Docker, API keys, target
model downloads, generation, or cache handling are required.

The classifier is intended for English and considers only the first 512 tokenizer
tokens. Pass the current request, not an entire long conversation. Empty prompts,
unknown modes, and invalid complexity thresholds fail before model loading.

Weights are under the NVIDIA Open Model License. `classifier.py` follows the
architecture/scoring in the official model card, uses strict safetensors loading,
and pins classifier revision `fea1121511eafabaf7dd6fc66863dcb04f74defb` and backbone
configuration revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.

## Tests

```sh
python3 -m unittest discover -s test -p 'test_routing.py' -v
src/venv/bin/python -m unittest discover -s test -p 'test_classifier.py' -v
```

Routing tests use fake predictions to check toggling, mappings, boundary scores,
validation, and CLI behavior without downloads. Classifier scoring tests use
synthetic logits and need the Python ML dependencies, but no weights or network.
