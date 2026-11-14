# Prompt → model choice

The router depends on the `Classifier` protocol in
[`interfaces.py`](interfaces.py), not a particular model or ML framework.
Implement `__call__(prompt: str) -> ClassificationResult`; no inheritance is
required. Return `task` for task routing, `complexity` for complexity routing,
or both. Optional confidence and dimension scores are available through
`analyze` but do not affect routing.

```python
from src.routing.interfaces import ClassificationResult
from src.routing.router import analyze, route

class MyClassifier:
    def __call__(self, prompt: str) -> ClassificationResult:
        # Replace with your classifier's prediction.
        return {"task": "general", "complexity": 0.2}

classifier = MyClassifier()
choice = route("Explain gravity", classifier_model=classifier,
               task_models={"general": "model_b", "technical": "model_a"})
choice = route("Explain gravity", mode="complexity",
               classifier_model=classifier, threshold=0.5)
print(analyze("Explain gravity", classifier_model=classifier))
```

- **task** (default): requires an explicit `task_models` mapping from your
  classifier's labels to answer-model aliases. Unmapped labels raise `KeyError`.
- **complexity**: selects `model_a` at or above `threshold`, otherwise `model_b`.
  Scores must be finite, with larger scores meaning more complex prompts.
  The threshold uses your classifier's scale; there is no fixed score range.
- Classifier inference failures log a warning and fall back to `model_a`.
  Invalid configuration and missing prediction fields raise errors.

Aliases default to `model_a` and `model_b`; standalone callers can override
these with the corresponding keyword arguments. Neither answer model is loaded
or called. Classifier construction and loading belong to the caller. The core
requires only Python's standard library; optional adapters are never imported
by the router or inference runner.

## Qwen/SmolLM/NVIDIA demo

[`demo_setup.py`](../experiments/poc/demo_setup.py) supplies the NVIDIA classifier
and the demo task mapping. `build_demo_runner()` additionally loads Qwen as
`model_a` and SmolLM as `model_b`, and accepts your KV cache mapper. The task
mapping sends Code Generation, Open QA, Other, and Unknown to Qwen; other labels
go to SmolLM. This mapping and the default threshold of 0.5 are experimental.

The routing CLI lives in the demo layer:

```sh
src/venv/bin/python -m src.experiments.poc.routing_exp --mode task "Write a Python function"
src/venv/bin/python -m src.experiments.poc.routing_exp --mode complexity --threshold 0.5 "Write a Python function"
```

See the [optional adapter documentation](../adapters/README.md) for dependencies,
checkpoint revisions, scoring, and licensing. Custom classifiers do not require
these dependencies.

## Tests

```sh
python3 -m unittest discover -s test -p 'test_routing.py' -v
src/venv/bin/python -m unittest discover -s test -p 'test_classifier.py' -v
```

Routing tests use fake predictions and check that the core works with Python
site-packages disabled. NVIDIA scoring tests use synthetic logits and need the
optional ML dependencies, but no weights or network.
