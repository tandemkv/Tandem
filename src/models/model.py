"""Greedy Instruct inference with explicit, copy-on-input cache handoff."""
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

class Model:
    def __init__(self, *, model_name, kv_cache=None, device="cpu",
                 dtype=torch.float32, revision=None):
        self.model_name = model_name
        self.revision = revision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, revision=revision
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.cache = None
        self.generated_tokens = []
        self._initial_cache = kv_cache

    def _validate_cache(self, cache):
        if not isinstance(cache, DynamicCache):
            raise ValueError("cache must be a transformers DynamicCache")
        owner = getattr(cache, "tandem_model_name", None)
        if owner is not None and owner != self.model_name:
            raise ValueError(f"cache belongs to {owner}, not {self.model_name}; translation required")
        config = self.model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        length = cache.get_seq_length()
        if len(cache.layers) != config.num_hidden_layers or length == 0:
            raise ValueError("incompatible cache: missing layers or empty history")
        expected = (1, config.num_key_value_heads, length, head_dim)
        for layer in cache.layers:
            for tensor in (layer.keys, layer.values):
                if (tensor is None or tuple(tensor.shape) != expected
                        or tensor.dtype != self.model.dtype or tensor.device != self.model.device
                        or not torch.isfinite(tensor).all()):
                    raise ValueError("incompatible cache: tensor shape, dtype, device or values")

    def run(self, prompt, *, kv_cache=None, is_suffix=False, max_new_tokens=100,
            prompt_mode="chat"):
        """Return answer text and expose the completed cache via get_kv_cache().

        Plain-text mode encodes verbatim with add_special_tokens=False.
        Ordinary chat prompts use the checkpoint's chat template. With a cache, pass
        only the uncached suffix and set is_suffix=True (include any required
        chat framing yourself).
        This avoids inserting a second system prompt into an existing chat.
        Explicit caches work on every call; a constructor cache is used once.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if prompt_mode not in ("chat", "plain"):
            raise ValueError("prompt_mode must be chat or plain")
        cache = kv_cache if kv_cache is not None else self._initial_cache
        if cache is not None and not is_suffix:
            raise ValueError("cached prompts must be an uncached suffix (is_suffix=True)")
        if is_suffix or prompt_mode == "plain":
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_dict=False
            )
        answer = self.run_tokens(tokens, kv_cache=cache, max_new_tokens=max_new_tokens)
        self._initial_cache = None
        return answer


    @torch.no_grad()
    def run_tokens(self, tokens, *, kv_cache=None, max_new_tokens=100):
        """Run exact uncached token IDs (including chat framing) after a cache.

        Useful for split-prefix replay without retokenizing a boundary. The
        caller is responsible for the provenance of untagged external caches.
        """
        if not tokens or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in tokens):
            raise ValueError("tokens must be a nonempty sequence of valid token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if kv_cache is not None:
            self._validate_cache(kv_cache)
        past = 0 if kv_cache is None else kv_cache.get_seq_length()
        if past + len(tokens) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt and generation exceed the model context window")
        cache = copy.deepcopy(kv_cache)
        output = self._forward(tokens, cache)
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        generated = []
        for _ in range(max_new_tokens):
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            # Include even the final token (including EOS) in the returned cache.
            output = self._forward([token], output.past_key_values)
            if token in eos:
                break
        self.cache = output.past_key_values
        self.cache.tandem_model_name = self.model_name
        self.generated_tokens = generated
        return self.tokenizer.decode(generated, skip_special_tokens=True)


    def _forward(self, tokens, cache):
        past = 0 if cache is None else cache.get_seq_length()
        device = self.model.device
        positions = torch.arange(past, past + len(tokens), device=device)
        return self.model(
            input_ids=torch.tensor([tokens], device=device),
            attention_mask=torch.ones((1, past + len(tokens)), dtype=torch.long, device=device),
            position_ids=positions[None, :], cache_position=positions,
            past_key_values=cache, use_cache=True,
        )


    def get_kv_cache(self):
        return self.cache
