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
CHECKPOINTS = {  # name: (dataset path, sha256, dataset revision)
    'ridge': (f'{RUN}/outputs/ridge-8000.pt', '84046a87b50ddfb3423ceece930ed7ca2b7055cda09d48b8435c91b74eb54193',
              REVISION),
    'distilled': (f'{RUN}/outputs/distilled-8000.pt',
                  'db99621579ef552721ffdf57b5308314d32e4944f87f2038f640ff65909bb0d6', REVISION),
    'keyfit': ('experiments/keyfit-20261124-190025/keyfit.pt',  # train_keys.py from distilled, 4000 steps
               'a3f59694adf0389482ff2871c8825833ad833cbd42c30fbda217653392de2d95',
               '14132052d3e7962c1718018e551be31061f85638'),
    'output': ('experiments/output-20261125-090640/output.pt',  # train_output.py from keyfit, 4000 steps
               '13e2a0a5c5dda74d2d002ca09a35133131eb7ed52833487f8af194581f2aff07',
               '9309f231d36c4265856d18d19748944deb6285ca'),
    'output2': ('experiments/output2-20261126-143528/output.pt',  # output continued on 11900 unseen prompts
                'eb6919b080a1e8f96c5a532db587421e0fbf3f0398031ac22050a827e4b6e151',
                '9a3f22df8ba069b957594531eaa341177a81b33f'),
    'output3': ('experiments/output3h0-20261127-051005/output.pt',  # output2, second pass of 4000 steps
                'aaf0ba9d48aab059a27873a9b4ef97af5a21b4063fc0a79fea676d7e95736ea7',
                '2321becd0b5c976ce8761734f16d6ce9e8255a3c'),
    'output3h5': ('experiments/output3h5-20261127-051005/output.pt',  # the same, with 5 native header tokens
                  '7f9bfcb779f26ce5428e92446387903a1de36e615b7bda7180c6ae6389e1ca47',
                  '62b7d01e4be495f4e6fc86fdbfdb5f323359673f'),
    'output4': ('experiments/output4-20261127-065421/output.pt',  # output3, rest of the second pass
                'f9a6cde1a2e6d25cd0b897f8a77701161468d0336b88ae276c336d9ca8c7dbc0',
                '58ccd341fadce9cd0d92fcf50a02e742df14135f'),
}
MAX_NEW_TOKENS = 32


def download(path, revision=REVISION, sha256=None):
    file = Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision))
    if sha256:
        with file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                raise ValueError(f'Checksum mismatch: {path}')
    return file


def split_rows(split='test'):
    """A pinned split ('train' or 'test'), verified against its manifest."""
    manifest = json.loads(download(f'{RUN}/data.json').read_text())
    with gzip.open(download(f'{RUN}/{split}.jsonl.gz'), 'rb') as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != manifest['sha256'][f'{split}.jsonl']:
        raise ValueError(f'{split} split did not match its manifest')
    return [json.loads(line) for line in data.splitlines()]


class Harness:
    """Both frozen models, their tokenizers, and the pinned mapper weights on one device."""

    def __init__(self, device='cuda', dtype=torch.float32):
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
        self.checkpoints = {name: load(path, revision=revision, sha256=digest)
                            for name, (path, digest, revision) in CHECKPOINTS.items()}
        eos = self.qwen.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])

    def forward(self, ids, cache=None):
        """Run Qwen on uncached token IDs after an optional cache; keep only last-position logits."""
        past = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(past, past + len(ids), device=self.device)
        return self.qwen(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache,
                         use_cache=True, logits_to_keep=1, cache_position=positions,
                         position_ids=positions[None], attention_mask=self.mask(cache, past + len(ids)))

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
                           attention_mask=self.mask(cache, past + len(ids))).logits[0, :-1].float()
        target = torch.tensor(tokens, device=self.device)
        return torch.nn.functional.cross_entropy(logits, target).item()

    def cache(self, features):
        """A fresh Qwen cache from unrotated K/V features, or None. Generation mutates caches.

        Features may come as (features, keep): prefix positions where keep is 0 get no attention.
        """
        if features is None:
            return None
        keep = None
        if isinstance(features, tuple):
            features, keep = features
        cache = to_cache(self.qwen, features)
        cache.keep = keep
        return cache

    def mask(self, cache, length):
        """The attention mask over cache and new tokens: ones, except prefix positions the cache drops."""
        mask = torch.ones((1, length), dtype=torch.long, device=self.device)
        keep = getattr(cache, 'keep', None)
        if keep is not None:
            mask[0, :len(keep)] = keep.to(self.device)
        return mask

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
