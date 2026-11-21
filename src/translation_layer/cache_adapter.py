"""Run Tandem's trained attention translator on a live Transformers cache."""
from bisect import bisect_right

import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

from src.translation_layer.interfaces import UnsupportedTranslationError


def _validate_model(model, spec, geometry):
    config = model.model.config
    shape = (config.num_hidden_layers, config.num_key_value_heads,
             getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
    if ((model.model_name, model.revision, config.model_type)
            != (spec["id"], spec["revision"], spec["type"]) or shape != geometry):
        raise ValueError("model, tokenizer revision or cache layout disagrees with the checkpoint")


def _split_tokens(tokenizer, text, boundary):
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tokens["input_ids"], tokens["offset_mapping"]
    prefix = tokenizer.encode(text[:boundary], add_special_tokens=False)
    count = len(prefix)
    if (not 0 < count < len(ids) or ids[:count] != prefix
            or offsets[count - 1][1] != boundary or offsets[count][0] < boundary):
        raise ValueError("handoff must end at a stable token boundary with an uncached suffix")
    return ids, offsets[:count], count


def _rotary(model, features, length):
    positions = torch.arange(length, device=features.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(features, positions)
    return cos.unsqueeze(1), sin.unsqueeze(1)


def _source_features(cache, model):
    length = cache.get_seq_length()
    cos, sin = _rotary(model, cache.layers[0].keys.float(), length)
    scale = cos.square() + sin.square()
    keys, values = [], []
    for layer in cache.layers:
        key = layer.keys.float()
        # Invert RoPE at the original source positions.
        unrotated = (key * cos - rotate_half(key) * sin) / scale
        keys.append(unrotated[0].transpose(0, 1))
        values.append(layer.values[0].float().transpose(0, 1))
    return torch.stack((torch.cat(keys, dim=1), torch.cat(values, dim=1)), dim=1)


def _translate_prefix(mapper, features, source_offsets, target_offsets, batch_size):
    # Match the offline mapper's latest shared token-ending character boundary.
    source_at_end = {end: i for i, (start, end) in enumerate(source_offsets) if end > start}
    shared = sorted(source_at_end.keys() & {end for start, end in target_offsets if end > start})
    target_rows, source_rows = [], []
    for row, (start, end) in enumerate(target_offsets):
        match = bisect_right(shared, end) - 1
        if start < end and match >= 0:
            target_rows.append(row)
            source_rows.append(source_at_end[shared[match]])

    # The offline mapper uses its training mean before the first causal match.
    output = mapper.target_mean.expand(len(target_offsets), -1, -1, -1).clone()
    for start in range(0, len(target_rows), batch_size):
        stop = start + batch_size
        output[target_rows[start:stop]] = mapper.predict(features[source_rows[start:stop]])
    return output


def _target_cache(features, model):
    config = model.config
    length = len(features)
    features = features.to(device=model.device, dtype=model.dtype)
    cos, sin = _rotary(model, features, length)
    shaped = features.reshape(length, 2, config.num_hidden_layers, config.num_key_value_heads, -1)
    cache = DynamicCache(config=config)
    for layer in range(config.num_hidden_layers):
        key = shaped[:, 0, layer].transpose(0, 1).unsqueeze(0)
        value = shaped[:, 1, layer].transpose(0, 1).unsqueeze(0).contiguous()
        cache.update(key * cos + rotate_half(key) * sin, value, layer)
    return cache


class CacheTranslationAdapter:
    """Translate live caches with caller-supplied, inference-ready translators.

    Pass (translator, checkpoint["run"]) pairs in translators. Each translator
    exposes config, predict(features), and target_mean on its inference device.
    """

    def __init__(self, *, models, translators):
        self.models = {model.model_name: model for model in models}
        self.directions = {}
        for mapper, run in translators:
            source, target = run["source"], run["target"]
            if {source, target} != {"smol", "qwen"}:
                raise ValueError("checkpoint must support a direction between SmolLM and Qwen")
            config = run["config"]
            if mapper.config != config:
                raise ValueError("translator configuration disagrees with the checkpoint")
            specs = run["models"][source], run["models"][target]
            pair = tuple(spec["id"] for spec in specs)
            if pair in self.directions:
                raise ValueError(f"duplicate checkpoint for {pair}")
            if run["batch"] < 1:
                raise ValueError("invalid checkpoint batch size")
            for role, spec in zip(("source", "target"), specs):
                model_type = {"smol": "llama", "qwen": "qwen2"}[run[role]]
                if spec["type"] != model_type:
                    raise ValueError("checkpoint uses an unsupported attention layout")
                _validate_model(self.models[spec["id"]], spec,
                                (config[f"{role}_layers"], config[f"{role}_heads"], config["head_dim"]))
            self.directions[pair] = mapper, run["batch"], specs[1]

    @torch.no_grad()
    def __call__(self, *, cache, source_model, target_model, conversation_text,
                 cached_token_ids, prefix_boundary, prompt_mode):
        pair = (source_model, target_model.model_name)
        if pair not in self.directions:
            raise UnsupportedTranslationError(*pair)
        if prompt_mode != "plain":
            raise ValueError("cache translation requires plain-text prompts")
        if type(prefix_boundary) is not int or not 0 < prefix_boundary < len(conversation_text):
            raise ValueError("handoff requires a nonempty prefix and suffix")

        mapper, batch_size, target_spec = self.directions[pair]
        config = mapper.config
        _validate_model(target_model, target_spec,
                        (config["target_layers"], config["target_heads"], config["head_dim"]))
        source = self.models[source_model]
        source._validate_cache(cache)
        if getattr(cache, "tandem_model_revision", source.revision) != source.revision:
            raise ValueError("source cache revision disagrees with the checkpoint")
        source_ids, source_offsets, source_length = _split_tokens(
            source.tokenizer, conversation_text, prefix_boundary)
        if (cached_token_ids != source_ids[:source_length] or cache.get_seq_length() != source_length
                or getattr(cache, "tandem_token_ids", cached_token_ids) != cached_token_ids):
            raise ValueError("source token IDs disagree with the cached prefix")
        target_ids, target_offsets, target_length = _split_tokens(
            target_model.tokenizer, conversation_text, prefix_boundary)

        features = _source_features(cache, source.model).to(mapper.target_mean)
        predicted = _translate_prefix(mapper, features, source_offsets, target_offsets, batch_size)
        translated = _target_cache(predicted, target_model.model)
        translated.tandem_model_name = target_model.model_name
        translated.tandem_model_revision = target_model.revision
        translated.tandem_token_ids = target_ids[:target_length]
        return translated, target_ids[target_length:]
