"""Native Qwen K/V for chosen prefix tokens: an oracle that copies them, and a deployable selective refresh.

Selections pick whole token columns (K and V, every layer), so K/V slot pairing and per-token layer
trajectories stay intact. `fractured` marks tokens whose span is not a shared tokenizer boundary,
`entity` marks mid-sentence capitalized words and numbers, `multi` marks words Qwen splits into pieces, and `random_like` draws a matched random set as a control.
"""
import random
import re

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def offsets(tokenizer, text, count):
    return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping[:count]


def widen(marked, count):
    """Add each marked token's neighbours."""
    return sorted({j for i in marked for j in (i - 1, i, i + 1) if 0 <= j < count})


def fractured(harness, prepared):
    text, count = prepared.case.prompt, len(prepared.target_ids)
    ends = {end for start, end in offsets(harness.tokenizers['smol'], text, len(prepared.source_ids)) if end > start}
    target = offsets(harness.tokenizers['qwen'], text, count)
    return widen([i for i, (start, end) in enumerate(target) if end > start and end not in ends], count)


def entity(harness, prepared):
    """Tokens of mid-sentence capitalized words and of words that hold a digit."""
    text, count = prepared.case.prompt, len(prepared.target_ids)
    target = offsets(harness.tokenizers['qwen'], text, count)
    spans = []
    for word in re.finditer(r"[^\W_][\w'-]*", text[:target[-1][1]]):
        before = text[:word.start()].rstrip(' ')
        opening = not before or before[-1] in '.!?\n:"('
        if (word.group()[0].isupper() and not opening) or any(c.isdigit() for c in word.group()):
            spans.append(word.span())
    return [i for i, (a, b) in enumerate(target) if b > a and any(a < e and s < b for s, e in spans)]


def multi(harness, prepared):
    """Every token of a word that Qwen splits into two or more pieces."""
    text, count = prepared.case.prompt, len(prepared.target_ids)
    target = offsets(harness.tokenizers['qwen'], text, count)
    marked = []
    for word in re.finditer(r"[^\W_][\w'-]*", text[:target[-1][1]]):
        pieces = [i for i, (a, b) in enumerate(target) if b > a and a < word.end() and word.start() < b]
        if len(pieces) > 1:
            marked.extend(pieces)
    return sorted(set(marked))


def body(select):
    """Drop the header line ("Passage 1:"): native header tokens act as a sink and confound the test."""
    def pick(harness, prepared):
        text, count = prepared.case.prompt, len(prepared.target_ids)
        header = text.find('\n') + 1
        starts = [start for start, end in offsets(harness.tokenizers['qwen'], text, count)]
        return [i for i in select(harness, prepared) if starts[i] >= header]
    return pick


def random_like(select):
    """A control: as many random prefix tokens as `select` marks, seeded by the case."""
    def pick(harness, prepared):
        text, count = prepared.case.prompt, len(prepared.target_ids)
        header = text.find('\n') + 1
        starts = [start for start, end in offsets(harness.tokenizers['qwen'], text, count)]
        pool = [i for i in range(count) if starts[i] >= header]
        return sorted(random.Random(prepared.case.id).sample(pool, len(select(harness, prepared))))
    return pick


fractured, entity, multi = body(fractured), body(entity), body(multi)
SELECTIONS = {'fractured': fractured, 'entity': entity, 'multi': multi,
              'random_fractured': random_like(fractured), 'random_entity': random_like(entity),
              'random_multi': random_like(multi)}


@torch.no_grad()
def refresh(qwen, ids, features, selected):
    """Recompute K/V for the selected prefix tokens with Qwen, in place over the translated features.

    One forward pass over the selected tokens only. At every layer they attend causally to the
    translated K/V of the other prefix tokens and to their own fresh K/V.
    """
    if not selected:
        return features
    model, config = qwen.model, qwen.config
    heads, groups = config.num_key_value_heads, config.num_attention_heads // config.num_key_value_heads
    features = features.clone()
    device = features.device
    positions = torch.tensor(selected, dtype=torch.long, device=device)
    every = torch.arange(len(ids), device=device)
    hidden = model.embed_tokens(torch.tensor([[ids[i] for i in selected]], dtype=torch.long, device=device))
    cos, sin = model.rotary_emb(hidden, every[None])
    mask = (every[None, :] <= positions[:, None])[None, None]
    for index, layer in enumerate(model.layers):
        attention, rows = layer.self_attn, slice(index * heads, (index + 1) * heads)
        x = layer.input_layernorm(hidden)
        shape = (1, len(selected), -1, attention.head_dim)
        q, k, v = (proj(x).view(shape).transpose(1, 2) for proj in (attention.q_proj, attention.k_proj, attention.v_proj))
        features[positions, 0, rows] = k[0].transpose(0, 1).to(features.dtype)
        features[positions, 1, rows] = v[0].transpose(0, 1).to(features.dtype)
        keys = features[:, 0, rows].transpose(0, 1)[None].to(q.dtype)
        values = features[:, 1, rows].transpose(0, 1)[None].to(q.dtype)
        _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, positions], sin[:, positions])
        out = torch.nn.functional.scaled_dot_product_attention(
            q, keys.repeat_interleave(groups, 1), values.repeat_interleave(groups, 1), attn_mask=mask)
        hidden = hidden + attention.o_proj(out.transpose(1, 2).reshape(1, len(selected), -1))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return features
