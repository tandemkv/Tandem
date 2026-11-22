# Tandem Source Code

- `experiments/`: cache replay experiments.
- [KV translation](experiments/kv_translation/README.md): offline SmolLM2-to-Qwen handoff and answer scoring.
- `experiments/poc/`: optional Qwen/SmolLM/NVIDIA demo wiring and entry points.
- [`routing/`](routing/README.md): model-independent routing using a supplied classifier and policy.
- [`adapters/`](adapters/README.md): optional model implementations and their dependencies.
- [`translation_layer/`](translation_layer/README.md): inference, cache handoff, and conversation management.
