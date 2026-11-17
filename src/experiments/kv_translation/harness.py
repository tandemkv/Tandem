"""Pinned models and data plus the shared Qwen forward, generation, and replay check."""
import gzip
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mapper import capture, to_cache

REPO = 'tandemkv/paired-kv-qwen-smollm'
REVISION = '27744a02d7d5f0abe61216612c46ec3b2a0bb52e'
RUN = 'experiments/ridge-32k.C70bPO'
PARENT_REVISION = 'ba8bfb17f27e5e6e10571093814c75a0c1fbef5c'
NORMALIZATION = ('experiments/continuation-ada-full-tokens-32-20261029/best.pt',
                 '713c5c140c01d42b26514025c4624a74d4493fa175243e4eb202a30c852371ad')
MODELS = {
    'smol': ('HuggingFaceTB/SmolLM2-360M-Instruct', 'a10cc1512eabd3dde888204e902eca88bddb4951'),
    'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775'),
}
CHECKPOINTS = {
    'ridge': ('ridge-8000.pt', '84046a87b50ddfb3423ceece930ed7ca2b7055cda09d48b8435c91b74eb54193'),
    'distilled': ('distilled-8000.pt', 'db99621579ef552721ffdf57b5308314d32e4944f87f2038f640ff65909bb0d6'),
}
MAX_NEW_TOKENS = 32


def download(path, revision=REVISION, sha256=None):
    file = Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision))
    if sha256:
        with file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                raise ValueError(f'Checksum mismatch: {path}')
    return file


def test_rows():
    """The pinned held-out split, verified against its manifest."""
    manifest = json.loads(download(f'{RUN}/data.json').read_text())
    with gzip.open(download(f'{RUN}/test.jsonl.gz'), 'rb') as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != manifest['sha256']['test.jsonl']:
        raise ValueError('Test split did not match its manifest')
    return [json.loads(line) for line in data.splitlines()]


class Harness:
    """Both frozen models, their tokenizers, and the pinned mapper weights on one device."""

    def __init__(self, device='cuda', dtype=torch.bfloat16):
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        self.device, self.dtype = device, dtype
        self.tokenizers = {k: AutoTokenizer.from_pretrained(m, revision=r) for k, (m, r) in MODELS.items()}
        self.models = {k: AutoModelForCausalLM.from_pretrained(
            m, revision=r, dtype=dtype, attn_implementation='sdpa').to(device).eval()
            for k, (m, r) in MODELS.items()}
        self.qwen, self.smol = self.models['qwen'], self.models['smol']
        load = lambda path, **kw: torch.load(download(path, **kw), map_location=device, weights_only=True)
        self.normalization = load(NORMALIZATION[0], revision=PARENT_REVISION, sha256=NORMALIZATION[1])['model']
        self.checkpoints = {name: load(f'{RUN}/outputs/{file}', sha256=digest)
                            for name, (file, digest) in CHECKPOINTS.items()}
        eos = self.qwen.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])

    def forward(self, ids, cache=None):
        """Run Qwen on uncached token IDs after an optional cache; keep only last-position logits."""
        past = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(past, past + len(ids), device=self.device)
        return self.qwen(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache,
                         use_cache=True, logits_to_keep=1, cache_position=positions,
                         position_ids=positions[None],
                         attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long, device=self.device))

    def generate(self, suffix, cache=None):
        """Greedy answer after the cache, with the first-step log-probabilities for diagnostics."""
        output = self.forward(suffix, cache)
        first = output.logits[0, -1].float().log_softmax(-1)
        tokens = []
        for _ in range(MAX_NEW_TOKENS):
            token = output.logits[0, -1].argmax().item()
            tokens.append(token)
            if token in self.eos or len(tokens) == MAX_NEW_TOKENS:
                break
            output = self.forward([token], output.past_key_values)
        text = self.tokenizers['qwen'].decode(tokens, skip_special_tokens=True).strip()
        return {'answer': text, 'tokens': tokens}, first

    def answer_nll(self, suffix, cache, tokens):
        """Mean negative log-likelihood of fixed answer tokens, teacher-forced after the cache."""
        ids = suffix + tokens
        past = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(past, past + len(ids), device=self.device)
        logits = self.qwen(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache,
                           use_cache=True, logits_to_keep=len(tokens) + 1, cache_position=positions,
                           position_ids=positions[None],
                           attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long,
                                                     device=self.device)).logits[0, :-1].float()
        target = torch.tensor(tokens, device=self.device)
        return torch.nn.functional.cross_entropy(logits, target).item()

    def cache(self, features):
        """A fresh Qwen cache from unrotated K/V features, or None. Generation mutates caches."""
        return None if features is None else to_cache(self.qwen, features)

    def check_replay(self, target_ids, suffix):
        """Native full prefill and native split-cache replay must pick the same greedy answer.

        Returns the native answer, its first-step log-probabilities, and its own answer NLL.
        """
        full, first = self.generate(target_ids + suffix)
        features = capture(self.qwen, target_ids)
        split, _ = self.generate(suffix, self.cache(features))
        if full['tokens'] != split['tokens']:
            raise AssertionError('Native split-cache replay diverged from full prefill')
        return full, first, self.answer_nll(suffix, self.cache(features), full['tokens'])
