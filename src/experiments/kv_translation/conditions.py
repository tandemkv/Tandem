"""Ways to build the Qwen prefix cache for a case. Each returns unrotated K/V features or None."""
import torch

from mapper import capture, overlap_alignment, split_prompt, translate
from refresh import FRACTIONS, SELECTIONS, body_tokens, refresh


class Prepared:
    """A case tokenized for both models, with the source prefix captured once."""

    def __init__(self, harness, case):
        self.case = case
        self.source_ids, self.target_ids, self.suffix, self.alignment = split_prompt(
            case.prompt, harness.tokenizers, case.split_at)
        self.source = capture(harness.smol, self.source_ids)
        self.overlap = overlap_alignment(case.prompt, harness.tokenizers, len(self.source_ids), len(self.target_ids))

    def alignment_for(self, checkpoint):
        return self.overlap if checkpoint.get('pooling') == 'overlap' else self.alignment


def question_only(harness, prepared):
    return None


def translated(name):
    def build(harness, prepared):
        checkpoint = harness.checkpoints[name]
        return translate(prepared.source, prepared.alignment_for(checkpoint), checkpoint,
                         harness.normalization, harness.smol.config.num_key_value_heads)
    return build


def permuted(name):
    """Control: translated features in a seeded random token order. Not the archived other-document swap."""
    def build(harness, prepared):
        features = translated(name)(harness, prepared)
        order = torch.randperm(len(features), generator=torch.Generator().manual_seed(0))
        return features[order.to(features.device)]
    return build


def with_native(role, translated_features, native_features):
    """Translated features with keys (role 0) or values (role 1) replaced by native Qwen's."""
    if translated_features.shape != native_features.shape:
        raise ValueError('Translated and native features differ in shape')
    mixed = translated_features.clone()
    mixed[:, role] = native_features[:, role]
    return mixed


def native_part(name, role):
    """Diagnostic: the translation with native Qwen keys or values, to locate where it loses quality."""
    def build(harness, prepared):
        native = capture(harness.qwen, prepared.target_ids)
        return with_native(role, translated(name)(harness, prepared), native)
    return build


def native_header(name, count):
    """The translation with its first tokens computed by Qwen itself, which causality makes exact and cheap."""
    def build(harness, prepared):
        features = translated(name)(harness, prepared).clone()
        features[:count] = capture(harness.qwen, prepared.target_ids[:count]).to(features.dtype)
        return features
    return build


def native_layers(name, first, last):
    """Diagnostic: the translation with native Qwen K/V in layers first..last-1, to locate the loss by depth."""
    def build(harness, prepared):
        heads = harness.qwen.config.num_key_value_heads
        mixed = translated(name)(harness, prepared).clone()
        native = capture(harness.qwen, prepared.target_ids)
        mixed[:, :, first * heads:last * heads] = native[:, :, first * heads:last * heads].to(mixed.dtype)
        return mixed
    return build


def column_oracle(name, selection):
    """Diagnostic: native Qwen K/V in every layer for the selected prefix tokens, translated elsewhere."""
    def build(harness, prepared):
        mixed = translated(name)(harness, prepared).clone()
        selected = torch.tensor(SELECTIONS[selection](harness, prepared), dtype=torch.long, device=mixed.device)
        mixed[selected] = capture(harness.qwen, prepared.target_ids)[selected].to(mixed.dtype)
        return mixed
    return build


def refreshed(name, selection):
    """Deployable: Qwen recomputes the selected prefix tokens over the translated cache of the rest."""
    def build(harness, prepared):
        return refresh(harness.qwen, prepared.target_ids, translated(name)(harness, prepared),
                       SELECTIONS[selection](harness, prepared))
    return build


def masked_oracle(name, selection):
    """Control: native K/V on the selected body tokens, the other body tokens masked out of attention.

    The same cache as column_oracle, so the only difference is translated versus absent for the rest.
    """
    def build(harness, prepared):
        features = column_oracle(name, selection)(harness, prepared)
        keep = torch.ones(len(features), dtype=torch.long)
        keep[body_tokens(harness, prepared)] = 0
        keep[SELECTIONS[selection](harness, prepared)] = 1
        return features, keep
    return build


def native(dtype=None):
    """Control: Qwen's own prefix K/V, optionally rounded through dtype, to bound what any map can reach."""
    def build(harness, prepared):
        features = capture(harness.qwen, prepared.target_ids)
        return features.to(dtype).to(features.dtype) if dtype else features
    return build


CONDITIONS = {
    'question_only': question_only,
    'native': native(),
    'native_bf16': native(torch.bfloat16),
    'ridge': translated('ridge'),
    'distilled': translated('distilled'),
    'keyfit': translated('keyfit'),
    'output': translated('output'),
    'output2': translated('output2'),
    'header1_output2': native_header('output2', 1),
    'header5_output2': native_header('output2', 5),
    'output3': translated('output3'),
    'output4': translated('output4'),
    'header5_output3': native_header('output3', 5),
    'header5_output3h5': native_header('output3h5', 5),
    'native_keys_output3': native_part('output3', 0),
    'native_values_output3': native_part('output3', 1),
    'native_early_output3': native_layers('output3', 0, 12),
    'native_late_output3': native_layers('output3', 12, 24),
    'oracle_fractured_output3': column_oracle('output3', 'fractured'),
    'oracle_entity_output3': column_oracle('output3', 'entity'),
    'oracle_random_fractured_output3': column_oracle('output3', 'random_fractured'),
    'oracle_random_entity_output3': column_oracle('output3', 'random_entity'),
    'oracle_multi_output3': column_oracle('output3', 'multi'),
    'oracle_random_multi_output3': column_oracle('output3', 'random_multi'),
    'refresh_fractured_output3': refreshed('output3', 'fractured'),
    'refresh_entity_output3': refreshed('output3', 'entity'),
    'refresh_random_fractured_output3': refreshed('output3', 'random_fractured'),
    'refresh_random_entity_output3': refreshed('output3', 'random_entity'),
    'refresh_multi_output3': refreshed('output3', 'multi'),
    'refresh_random_multi_output3': refreshed('output3', 'random_multi'),
    **{f'{kind}_{name}{p}_output3': build('output3', f'{name}{p}') for p in FRACTIONS
       for kind, build in (('oracle', column_oracle), ('refresh', refreshed))
       for name in ('scattered', 'lead', 'hub')},
    **{f'masked_scattered{p}_output3': masked_oracle('output3', f'scattered{p}') for p in FRACTIONS},
    'masked_scattered0_output3': masked_oracle('output3', 'scattered0'),
    'permuted_distilled': permuted('distilled'),
    'native_keys_distilled': native_part('distilled', 0),
    'native_values_distilled': native_part('distilled', 1),
}
