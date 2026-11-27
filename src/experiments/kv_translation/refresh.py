"""Native Qwen K/V for chosen prefix tokens: an oracle that copies them, and a deployable selective refresh.

Selections pick whole token columns (K and V, every layer), so K/V slot pairing and per-token layer
trajectories stay intact. `fractured` marks tokens whose span is not a shared tokenizer boundary,
`entity` marks mid-sentence capitalized words and numbers, `multi` marks words Qwen splits into pieces, and `random_like` draws a matched random set as a control.
`scattered`, `lead` and `hub` pick a share of the body: random tokens, its start, or the contiguous window
SmolLM's last prefix token attends to most.
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


def body_tokens(harness, prepared):
    """Prefix tokens after the header line."""
    text, count = prepared.case.prompt, len(prepared.target_ids)
    header = text.find('\n') + 1
    return [i for i, (start, end) in enumerate(offsets(harness.tokenizers['qwen'], text, count)) if start >= header]


def scattered(fraction):
    """A random share of the body tokens, seeded by the case."""
    def pick(harness, prepared):
        tokens = body_tokens(harness, prepared)
        return sorted(random.Random(prepared.case.id).sample(tokens, round(fraction * len(tokens))))
    return pick


def lead(fraction):
    """The first share of the body: the title-subject sentence and what follows it."""
    def pick(harness, prepared):
        tokens = body_tokens(harness, prepared)
        return tokens[:round(fraction * len(tokens))]
    return pick


@torch.no_grad()
def salience(harness, prepared, layers=range(16, 29)):
    """Per target token: how much SmolLM's last prefix token attends to its aligned source token."""
    smol, queries, handles = harness.smol, [], []
    for index in layers:
        handles.append(smol.model.layers[index].self_attn.q_proj.register_forward_hook(
            lambda _, __, out: queries.append(out[0, -1])))
    try:
        smol(input_ids=torch.tensor([prepared.source_ids], device=smol.device), use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    config, count = smol.config, len(prepared.source_ids)
    heads, dim = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    cos, sin = smol.model.rotary_emb(prepared.source, torch.arange(count, device=smol.device)[None])
    scores = torch.zeros(count, device=smol.device)
    for q, index in zip(queries, layers):
        keys = prepared.source[:, 0, index * heads:(index + 1) * heads].float().transpose(0, 1)[None]
        _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
        q = q.float().view(1, -1, 1, dim)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
        keys = keys.repeat_interleave(q.shape[1] // heads, dim=1)
        scores += ((q @ keys.transpose(-1, -2))[0, :, 0] / dim ** 0.5).softmax(-1).mean(0)
    return [scores[a].item() if a >= 0 else 0.0 for a in prepared.alignment]


def hub(fraction):
    """The contiguous body window of that share that SmolLM's last prefix token attends to most."""
    def pick(harness, prepared):
        tokens, score = body_tokens(harness, prepared), salience(harness, prepared)
        width = round(fraction * len(tokens))
        if not width:
            return []
        sums = [sum(score[i] for i in tokens[j:j + width]) for j in range(len(tokens) - width + 1)]
        best = max(range(len(sums)), key=sums.__getitem__)
        return tokens[best:best + width]
    return pick


fractured, entity, multi = body(fractured), body(entity), body(multi)
SELECTIONS = {'fractured': fractured, 'entity': entity, 'multi': multi,
              'random_fractured': random_like(fractured), 'random_entity': random_like(entity),
              'random_multi': random_like(multi)}
FRACTIONS = (15, 33, 50, 75)
SELECTIONS['scattered0'] = scattered(0.0)
SELECTIONS['scattered100'] = scattered(1.0)


def sentence(harness, prepared):
    """The body up to the end of its first sentence after the title line."""
    text, tokens = prepared.case.prompt, body_tokens(harness, prepared)
    starts = [start for start, end in offsets(harness.tokenizers['qwen'], text, len(prepared.target_ids))]
    title = text.find('\n', text.find('\n') + 1) + 1
    end = min((i for i in (text.find('. ', title), text.find('.\n', title)) if i >= 0), default=len(text)) + 1
    return [i for i in tokens if starts[i] < end]


def with_header(select):
    """The selection plus the header line, so a lead selection is one contiguous native prefix."""
    def pick(harness, prepared):
        body = set(body_tokens(harness, prepared))
        header = [i for i in range(len(prepared.target_ids)) if i not in body]
        return sorted(set(header) | set(select(harness, prepared)))
    return pick


SELECTIONS |= {f'{name}{p}': make(p / 100) for p in FRACTIONS
               for name, make in (('scattered', scattered), ('lead', lead), ('hub', hub))}


SELECTIONS['sentence'] = sentence
SELECTIONS |= {f'h{name}': with_header(SELECTIONS[name])
               for name in ('scattered75', 'scattered100', 'lead33', 'sentence')}


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
