# Semantic routing for the proof of concept

For Tandem's proof of concept, we chose a small, custom semantic router that selects between SmolLM and Qwen. The router returns a model identifier, while our inference code remains responsible for running the selected model and managing its KV cache. This keeps the routing decision explicit and gives us direct control over the execution and cache behavior we want to study.

Production routing systems often offer a broader request-to-response workflow. For example, the documented vLLM Semantic Router integration with vLLM Production Stack classifies an OpenAI-compatible request, selects a backend, and forwards the request through the serving infrastructure. It also supports features such as semantic caching and safety checks. These capabilities are useful for deployment, but introduce integration work beyond the model-selection function needed for our initial experiment. See the [official integration guide](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/semantic-router-integration.html).

There is an important qualification: semantic routing does not inherently require owning model execution, and vLLM Semantic Router is a decision layer between clients and model backends. It is therefore too strong to say that existing routers necessarily bundle the models themselves or prevent custom inference. Our reason for choosing a custom implementation is the smaller integration scope and direct access to local inference state, rather than a fundamental limitation of semantic routing. The project's [system overview](https://vllm-sr.ai/docs/overview/semantic-router-overview/) describes this separation between routing and backends.

In particular, semantic response caching and a model's KV cache serve different purposes. A semantic response cache can reuse an answer for a similar request; a KV cache stores attention keys and values used during model inference. Tandem needs control at the latter level: supplying a compatible cache to generation, inspecting the resulting cache, and deciding when to reuse it. Our current Transformers/PyTorch model wrapper accepts an initial cache through `past_key_values` and exposes the cache returned by generation. This access does not, by itself, make caches interchangeable between SmolLM and Qwen; cross-model cache compatibility or conversion is a separate concern.

The proof-of-concept routing path is deliberately simple:

1. NVIDIA's pretrained prompt task and complexity classifier analyzes the prompt.
2. A configurable task-to-model mapping, or a complexity threshold, selects SmolLM or Qwen.
3. The caller uses that selection to run the target model through our Transformers/PyTorch inference code, with explicit cache handling.

The implementation in `src/routing/router.py` keeps classification separate from target-model inference: `analyze()` exposes the classifier's scores, while `route()` returns `"smollm"` or `"qwen"`. Unknown classifications and classifier loading or inference failures default to Qwen. The task mapping and complexity threshold are experimental policies, not evidence that one model is better for every task assigned to it; they need validation against measured quality and execution cost.

This approach lets us evaluate the two-model design without first adapting a production serving stack or investigating whether changes to vLLM Semantic Router would be necessary. A fork is a possible integration cost, not an established requirement. Once the proof of concept demonstrates useful behavior, we can assess production routing infrastructure against concrete requirements for inference, cache access, latency, and deployment.
