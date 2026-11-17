"""Ways to build the Qwen prefix cache for a case. Each returns unrotated K/V features or None."""
import torch

from mapper import capture, split_prompt, translate


class Prepared:
    """A case tokenized for both models, with the source prefix captured once."""

    def __init__(self, harness, case):
        self.case = case
        self.source_ids, self.target_ids, self.suffix, self.alignment = split_prompt(
            case.prompt, harness.tokenizers)
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


CONDITIONS = {
    'question_only': question_only,
    'ridge': translated('ridge'),
    'distilled': translated('distilled'),
    'permuted_distilled': permuted('distilled'),
}
