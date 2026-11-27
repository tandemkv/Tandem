"""Load the frozen affine K/V map and keep the question out of the source cache."""
from bisect import bisect_right

import torch
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def align(source_offsets, target_offsets):
    """Use the most recent shared character boundary, or -1 for the mean prior."""
    source = {end: i for i, (start, end) in enumerate(source_offsets) if end > start}
    shared = sorted(source.keys() & {end for start, end in target_offsets if end > start})
    result = []
    for start, end in target_offsets:
        i = bisect_right(shared, end) - 1
        result.append(source[shared[i]] if i >= 0 and end > start else -1)
    return result



def overlap(source_offsets, target_offsets):
    """Each target token's source tokens, weighted by the share of its characters each one covers."""
    rows, first = [], 0
    for start, end in target_offsets:
        while first < len(source_offsets) and source_offsets[first][1] <= start:
            first += 1
        row, i = [], first
        while end > start and i < len(source_offsets) and source_offsets[i][0] < end:
            covered = min(end, source_offsets[i][1]) - max(start, source_offsets[i][0])
            if covered > 0:
                row.append((i, covered))
            i += 1
        total = sum(c for _, c in row)
        rows.append([(i, c / total) for i, c in row])
    return rows


def overlap_alignment(text, tokenizers, source_count, target_count):
    """Overlap weights for the first source_count and target_count tokens of text."""
    offsets = {name: tok(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
               for name, tok in tokenizers.items()}
    return overlap(offsets['smol'][:source_count], offsets['qwen'][:target_count])


def pool(source, rows):
    """Source features pooled onto target tokens by overlap weights. Tokens with no overlap get -1."""
    weights = torch.zeros(len(rows), len(source), device=source.device)
    for t, row in enumerate(rows):
        for i, w in row:
            weights[t, i] = w
    pooled = (weights @ source.flatten(1).float()).view(len(rows), *source.shape[1:]).to(source.dtype)
    return pooled, [t if row else -1 for t, row in enumerate(rows)]


def split_prompt(text, tokenizers, marker='\n\nQuestion: '):
    """Split at the last shared token boundary before the marker. The prefix is translated."""
    boundary = text.rfind(marker)
    if boundary < 1:
        raise ValueError(f'Prompt has no {marker!r} boundary')
    encodings = {name: tok(text, add_special_tokens=False, return_offsets_mapping=True)
                 for name, tok in tokenizers.items()}
    ends = {name: {end: i + 1 for i, (start, end) in reversed(list(enumerate(e.offset_mapping)))
                   if end > start} for name, e in encodings.items()}
    shared = ends['smol'].keys() & ends['qwen'].keys()
    cut = max((end for end in shared if end <= boundary), default=0)
    if not cut:
        raise ValueError('Prompt has no shared context boundary')
    prefixes, offsets = {}, {}
    for name, encoded in encodings.items():
        count = ends[name][cut]
        prefixes[name] = encoded.input_ids[:count]
        offsets[name] = encoded.offset_mapping[:count]
        if prefixes[name] != tokenizers[name].encode(text[:cut], add_special_tokens=False):
            raise ValueError(f'Prefix retokenization changed {name} IDs')
    suffix = encodings['qwen'].input_ids[len(prefixes['qwen']):]
    return prefixes['smol'], prefixes['qwen'], suffix, align(offsets['smol'], offsets['qwen'])


def capture(model, ids):
    """Capture unrotated K/V as [token, K/V, layer-head, head dimension]."""
    keys, values, handles = [], [], []

    def save(outputs):
        def hook(_, __, tensor):
            outputs.append(tensor[0])
        return hook

    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, 'k_norm'):
            raise ValueError('This mapper expects keys without key normalization')
        handles.extend((attention.k_proj.register_forward_hook(save(keys)),
                        attention.v_proj.register_forward_hook(save(values))))
    try:
        model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    heads = model.config.num_key_value_heads
    return torch.stack([torch.stack([t.reshape(len(ids), heads, -1) for t in role], dim=1).flatten(1, 2)
                        for role in (keys, values)], dim=1)


def translate(source, alignment, checkpoint, normalization, source_heads):
    source_mean, source_std, target_mean, target_std = (
        normalization[k] for k in ('source_mean', 'source_std', 'target_mean', 'target_std')
    )
    if alignment and isinstance(alignment[0], list):  # Overlap rows: pool the source onto target tokens.
        source, alignment = pool(source, alignment)
    if any(position < -1 or position >= len(source) for position in alignment):
        raise ValueError('Alignment points outside the source cache')
    output = target_mean.expand(len(alignment), *target_mean.shape).clone()
    indices = torch.tensor(alignment, dtype=torch.long, device=source.device)
    valid = (indices >= 0).nonzero().flatten()
    for batch in valid.split(1024):  # The map runs in float32 even when the models use bfloat16.
        x = (source[indices[batch]].float() - source_mean) / source_std
        x = x.unflatten(-2, (-1, source_heads))[:, :, checkpoint['layers']].flatten(-3)
        if checkpoint.get('coupled'):  # Each of K and V reads both: its own features, then the other's.
            x = torch.cat((x, x.flip(1)), dim=-1)
        x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        y = torch.einsum('brli,rlio->brlo', x, checkpoint['weights'])
        if 'up' in checkpoint:  # A nonlinear residual per role and layer, on top of the affine map.
            hidden = torch.nn.functional.gelu(torch.einsum('brli,rlih->brlh', x, checkpoint['up']))
            y = y + torch.einsum('brlh,rlho->brlo', hidden, checkpoint['down'])
        output[batch] = y.reshape(-1, *target_mean.shape) * target_std + target_mean
    return output.to(source.dtype)


def to_cache(model, features):
    heads = model.config.num_key_value_heads
    positions = torch.arange(len(features), device=features.device)[None]
    cos, sin = model.model.rotary_emb(features, positions)
    cache = DynamicCache(config=model.config)
    shaped = features.reshape(len(features), 2, model.config.num_hidden_layers, heads, -1)
    for layer in range(model.config.num_hidden_layers):
        key, value = [part.permute(1, 0, 2).unsqueeze(0).contiguous()
                      for part in shaped[:, :, layer].unbind(1)]
        _, key = apply_rotary_pos_emb(key, key, cos, sin)
        cache.update(key, value, layer)
    return cache
