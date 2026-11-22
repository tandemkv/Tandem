"""Plain-text SmolLM2 → Qwen2.5 adapter for the distilled-8000 affine map."""
from bisect import bisect_right
import hashlib

import torch
from huggingface_hub import hf_hub_download
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import rotate_half

from src.translation_layer.interfaces import UnsupportedTranslationError

SOURCE = ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951")
TARGET = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
REPO = "tandemkv/paired-kv-qwen-smollm"


def _load(path, revision, digest):
    filename = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
    with open(filename, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"Checkpoint checksum mismatch for {path}")
        stream.seek(0)
        return torch.load(stream, weights_only=True, map_location="cpu")


def _split(source, target, text, boundary, cached_ids):
    if not 0 < boundary < len(text) or source.encode(text[:boundary], add_special_tokens=False) != cached_ids:
        raise ValueError("Source token IDs must represent the exact plain-text prefix")
    encodings = [tok(text, add_special_tokens=False, return_offsets_mapping=True)
                 for tok in (source, target)]
    # Match the offline split at the last complete boundary shared by both tokenizers.
    ends = [{end: i + 1 for i, (start, end) in reversed(list(enumerate(e.offset_mapping)))
             if start < end <= boundary} for e in encodings]
    shared = sorted(ends[0].keys() & ends[1].keys())
    if not shared:
        raise ValueError("No shared token boundary in the cached prefix")
    cut = shared[-1]
    counts = [mapping[cut] for mapping in ends]
    src, tgt = encodings
    for tok, encoded, count in zip((source, target), encodings, counts):
        if encoded.input_ids[:count] != tok.encode(text[:cut], add_special_tokens=False):
            raise ValueError("Unstable tokenization at the shared handoff boundary")
    if src.input_ids[:counts[0]] != cached_ids[:counts[0]] or counts[1] == len(tgt.input_ids):
        raise ValueError("Handoff requires an exact cached prefix and a target suffix")
    # Repeated offsets (e.g. UTF-8 byte tokens) follow the offline aligner's
    # last source index, restricted to tokens actually retained at the cut.
    positions = {end: i for i, (start, end) in enumerate(src.offset_mapping[:counts[0]]) if end > start}
    alignment = []
    for start, end in tgt.offset_mapping[:counts[1]]:
        i = bisect_right(shared, end) - 1
        alignment.append(positions[shared[i]] if i >= 0 and end > start else -1)
    return alignment, tgt.input_ids[:counts[1]], tgt.input_ids[counts[1]:]


class DistilledKVMapper:
    """Load the pinned checkpoint once for an existing pair of Tandem Models."""

    def __init__(self, source, target):
        if (source.model_name, source.revision) != SOURCE or (target.model_name, target.revision) != TARGET:
            raise ValueError("Distilled translation requires the pinned SmolLM and Qwen model/tokenizer revisions")
        self.source, self.target = source, target
        device = target.model.device
        checkpoint = _load("experiments/ridge-32k.C70bPO/outputs/distilled-8000.pt",
                           "27744a02d7d5f0abe61216612c46ec3b2a0bb52e",
                           "db99621579ef552721ffdf57b5308314d32e4944f87f2038f640ff65909bb0d6")
        norms = _load("experiments/continuation-ada-full-tokens-32-20261029/best.pt",
                      "ba8bfb17f27e5e6e10571093814c75a0c1fbef5c",
                      "713c5c140c01d42b26514025c4624a74d4493fa175243e4eb202a30c852371ad")["model"]
        self.layers = torch.as_tensor(checkpoint["layers"], device=device)
        self.weights = checkpoint["weights"].to(device=device, dtype=torch.float32)
        for role, model in (("source", source.model), ("target", target.model)):
            cfg = model.config
            dim = cfg.hidden_size // cfg.num_attention_heads
            for stat in ("mean", "std"):
                name = f"{role}_{stat}"
                tensor = norms[name]
                if tensor.shape != (2, cfg.num_hidden_layers * cfg.num_key_value_heads, dim):
                    raise ValueError(f"Incompatible {name} feature layout")
                setattr(self, name, tensor.to(device=device, dtype=torch.float32))
        sc, tc = source.model.config, target.model.config
        if (self.layers.ndim != 2 or self.layers.shape[0] != tc.num_hidden_layers
                or self.layers.dtype != torch.long or self.layers.min() < 0
                or self.layers.max() >= sc.num_hidden_layers):
            raise ValueError("Incompatible checkpoint source-layer selection")
        inputs = self.layers.shape[1] * sc.num_key_value_heads * self.source_mean.shape[-1]
        outputs = tc.num_key_value_heads * self.target_mean.shape[-1]
        if self.weights.shape != (2, tc.num_hidden_layers, inputs + 1, outputs):
            raise ValueError("Incompatible affine weight layout")

    @torch.no_grad()
    def __call__(self, *, cache, source_model, target_model, conversation_text,
                 cached_token_ids, prefix_boundary, prompt_mode):
        if (source_model, target_model.model_name) != (SOURCE[0], TARGET[0]):
            raise UnsupportedTranslationError(source_model, target_model.model_name)
        if prompt_mode != "plain" or target_model is not self.target:
            raise ValueError("Use plain text and the Qwen instance supplied to the mapper")
        if (getattr(cache, "tandem_model_revision", None) != SOURCE[1]
                or getattr(cache, "tandem_model_name", None) != SOURCE[0]
                or getattr(cache, "tandem_token_ids", None) != cached_token_ids
                or cache.get_seq_length() != len(cached_token_ids)):
            raise ValueError("Source cache provenance or token IDs do not match")
        self.source._validate_cache(cache)
        alignment, prefix, suffix = _split(self.source.tokenizer, target_model.tokenizer,
                                          conversation_text, prefix_boundary, cached_token_ids)
        source = self.source.model
        keys = torch.stack([layer.keys[0] for layer in cache.layers])  # layer, head, token, dim
        values = torch.stack([layer.values[0] for layer in cache.layers])
        positions = torch.arange(keys.size(2), device=keys.device)[None]
        cos, sin = source.model.rotary_emb(keys, positions)
        # SmolLM's live keys already include RoPE. The map was fit to pre-RoPE projections.
        keys = keys * cos - rotate_half(keys) * sin
        features = torch.stack((keys, values)).permute(3, 0, 1, 2, 4).flatten(2, 3)
        features = features.to(self.weights.device)
        output = self.target_mean.expand(len(prefix), *self.target_mean.shape).clone()
        indices = torch.tensor(alignment, device=features.device)
        for batch in (indices >= 0).nonzero().flatten().split(1024):
            if not batch.numel():
                continue
            x = (features[indices[batch]].float() - self.source_mean) / self.source_std
            x = x.unflatten(-2, (-1, source.config.num_key_value_heads))[:, :, self.layers].flatten(-3)
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
            mapped = torch.einsum("brli,rlio->brlo", x, self.weights)
            output[batch] = mapped.reshape(-1, *self.target_mean.shape) * self.target_std + self.target_mean

        target = target_model.model
        cfg = target.config
        shaped = output.to(target.dtype).reshape(len(prefix), 2, cfg.num_hidden_layers, cfg.num_key_value_heads, -1)
        cos, sin = target.model.rotary_emb(shaped, torch.arange(len(prefix), device=target.device)[None])
        translated = DynamicCache(config=cfg)
        for layer in range(cfg.num_hidden_layers):
            key, value = (part.transpose(0, 1).unsqueeze(0) for part in shaped[:, :, layer].unbind(1))
            translated.update(key * cos + rotate_half(key) * sin, value.contiguous(), layer)
        translated.tandem_model_name, translated.tandem_model_revision = TARGET
        translated.tandem_token_ids = prefix
        return translated, suffix
