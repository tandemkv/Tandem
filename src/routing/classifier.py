"""NVIDIA's multi-head DeBERTa classifier, loaded from pinned safetensors.

Architecture and scoring follow the official model card:
https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
Weights use the NVIDIA Open Model License, not the repository's code license.
"""
import json

import torch
from torch import nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "nvidia/prompt-task-and-complexity-classifier"
REVISION = "fea1121511eafabaf7dd6fc66863dcb04f74defb"
BACKBONE = "microsoft/deberta-v3-base"
BACKBONE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
COMPLEXITY_WEIGHTS = {
    "creativity_scope": 0.35,
    "reasoning": 0.25,
    "constraint_ct": 0.15,
    "domain_knowledge": 0.15,
    "contextual_knowledge": 0.05,
    "number_of_few_shots": 0.05,
}


class _Network(nn.Module):
    def __init__(self, config, backbone_config):
        super().__init__()
        # Construct architecture only; classifier checkpoint supplies all weights.
        self.backbone = AutoModel.from_config(backbone_config)
        self.targets = tuple(config["target_sizes"])
        for index, size in enumerate(config["target_sizes"].values()):
            head = nn.Module()
            head.add_module("fc", nn.Linear(backbone_config.hidden_size, size))
            self.add_module(f"head_{index}", head)

    def forward(self, tokens):
        hidden = self.backbone(input_ids=tokens["input_ids"],
                               attention_mask=tokens["attention_mask"]).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return {target: getattr(self, f"head_{index}").fc(pooled)[0]
                for index, target in enumerate(self.targets)}


def decode(logits, config):
    """Apply the publisher's class weights, normalization and rounding."""
    probabilities = logits["task_type"].softmax(-1)
    task_id = int(probabilities.argmax())
    dimensions = {}
    for target in COMPLEXITY_WEIGHTS:
        probabilities_for_target = logits[target].softmax(-1)
        weights = probabilities_for_target.new_tensor(config["weights_map"][target])
        score = float((probabilities_for_target * weights).sum()) / config["divisor_map"][target]
        score = round(score, 4)
        if target == "number_of_few_shots" and score < 0.05:
            score = 0
        dimensions[target] = score
    complexity = round(sum(dimensions[key] * weight
                           for key, weight in COMPLEXITY_WEIGHTS.items()), 5)
    return {"task": config["task_type_map"][str(task_id)],
            "task_confidence": round(float(probabilities[task_id]), 3),
            "complexity": complexity, "dimensions": dimensions}


class PromptClassifier:
    def __init__(self):
        config_path = hf_hub_download(MODEL, "config.json", revision=REVISION)
        with open(config_path, encoding="utf-8") as source:
            self.config = json.load(source)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        backbone_config = AutoConfig.from_pretrained(BACKBONE, revision=BACKBONE_REVISION)
        self.model = _Network(self.config, backbone_config)
        checkpoint = hf_hub_download(MODEL, "model.safetensors", revision=REVISION)
        load_model(self.model, checkpoint, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, prompt):
        tokens = self.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        return decode(self.model(tokens), self.config)
