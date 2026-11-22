# Tandem routing demo

From the repository root, set up the experiment environment and start the interactive session:

```sh
uv venv src/venv
uv pip install --python src/venv/bin/python torch transformers==5.20.1 huggingface-hub safetensors sentencepiece protobuf
src/venv/bin/python -m src.experiments.poc
```

Reuse an existing experiment environment by skipping the environment creation
step. Run once with internet access to cache the classifier, tokenizer files
and both answer models. With the files cached, set `HF_HUB_OFFLINE=1` to
rehearse offline.

At `You>`, enter a prompt and press Return. The demo prints the task, model,
and answer, then waits for another prompt. Type `/quit` or press Ctrl-D to exit.
Try these prompts in sequence to exercise same-model context reuse:

- Write a one-line Python function that squares a number. Return only the code.
- What about in C++?

For a scripted run with independent prompts, pass each prompt as a quoted
command-line argument. Interactive turns share a conversation. Continuing it across
models requires a compatible KV-cache mapper and trained checkpoint for the
switch direction (see issues #73 and #74).
