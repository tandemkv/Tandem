"""Ways to build the Qwen prefix cache for a case. Each returns unrotated K/V features or None."""
import torch

from mapper import capture, split_prompt, translate


class Prepared:
    """A case tokenized for both models, with the source prefix captured once."""

    def __init__(self, harness, case):
        self.case = case
        self.source_ids, self.target_ids, self.suffix, self.alignment = split_prompt(
            case.prompt, harness.tokenizers, case.split_at)
        self.source = capture(harness.smol, self.source_ids)


def question_only(harness, prepared):
    return None


def translated(name):
    def build(harness, prepared):
        return translate(prepared.source, prepared.alignment, harness.checkpoints[name],
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


CONDITIONS = {
    'question_only': question_only,
    'ridge': translated('ridge'),
    'distilled': translated('distilled'),
    'keyfit': translated('keyfit'),
    'permuted_distilled': permuted('distilled'),
    'native_keys_distilled': native_part('distilled', 0),
    'native_values_distilled': native_part('distilled', 1),
}
