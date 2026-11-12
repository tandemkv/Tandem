# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2",
#   "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5",
# ]
# ///
"""Prepare, extract and validate paired pre-RoPE K / projected V features."""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import itertools
import json
import os
import platform
import random
import resource
import sys
import time
import unicodedata
from pathlib import Path

import certifi

os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen": {"id": "Qwen/Qwen2.5-0.5B-Instruct",
             "revision": "7ae557604adf67be50417f59c2c2f167def9a775", "type": "qwen2"},
    "smol": {"id": "HuggingFaceTB/SmolLM2-360M-Instruct",
             "revision": "a10cc1512eabd3dde888204e902eca88bddb4951", "type": "llama"},
}
SOURCE = {"id": "hotpotqa/hotpot_qa",
          "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
          "file": "distractor/train-00000-of-00002.parquet",
          "license": "CC-BY-SA-4.0", "url": "https://hotpotqa.github.io/"}
SPACE = "k_proj_before_key_norm_and_rope__v_proj__token_major_fp32_v1"
EXPECTED_GEOMETRY = {
    "qwen": {"layers": 24, "kv_heads": 2, "head_dim": 64, "width": 128},
    "smol": {"layers": 32, "kv_heads": 5, "head_dim": 64, "width": 320},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def document_key(title):
    return sha(unicodedata.normalize("NFKC", title).casefold().strip().encode())


def format_prompt(row):
    titles, bodies = row["context"]["title"], row["context"]["sentences"]
    if not titles or len(titles) != len(bodies):
        raise ValueError("Invalid context")
    passages = [f"Passage {i + 1}:\n{title}\n{''.join(body).strip()}"
                for i, (title, body) in enumerate(zip(titles, bodies))]
    prefix = "\n\n".join(passages) + "\n\nQuestion: "
    question = row["question"].strip()
    if not question:
        raise ValueError("Empty question")
    return prefix + question + "\nAnswer:", [len(prefix), len(prefix) + len(question)]


def encode(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [list(pair) for pair in encoded["offset_mapping"]]
    if not ids or len(ids) != len(offsets):
        raise ValueError("Invalid tokenization")
    if any(not 0 <= a <= b <= len(text) for a, b in offsets):
        raise ValueError("Offsets outside raw text")
    if any(a[1] > b[1] for a, b in itertools.pairwise(offsets)):
        raise ValueError("Token ends are not monotonic")
    return {"ids": ids, "offsets": offsets}


def align(source_offsets, target_offsets):
    # Match common *character-end boundaries*, never token index alone.
    source = {b: i for i, (a, b) in enumerate(source_offsets) if b > a}
    target = {b for a, b in target_offsets if b > a}
    common = sorted(source.keys() & target)
    if len(common) < 8:
        raise ValueError("Fewer than eight shared token boundaries")
    first = next(i for i, (a, b) in enumerate(source_offsets) if b > a)
    mapping, pairs, modes = [], [], []
    for j, (a, b) in enumerate(target_offsets):
        index = bisect.bisect_right(common, b) - 1
        mapped = source[common[index]] if index >= 0 else first
        mapping.append(mapped)
        exact = b > a and b in source
        if exact:
            pairs.append([mapped, j])
        modes.append("exact" if exact else "forward_fill" if index >= 0 else "before_first")
    return {"target_to_source": mapping, "exact_pairs": pairs, "modes": modes}


def prepare(args):
    import pyarrow.parquet as pq
    if (args.output / "manifest.json").exists():
        raise ValueError("Manifest already exists; use a new output directory")
    if args.count < 2 or args.scan_rows < args.count:
        raise ValueError("Need at least two examples and sufficient scan rows")
    if not 64 <= args.max_tokens <= 8192:
        raise ValueError("Token limit must fit both models: 64..8192")
    tokenizers = {k: AutoTokenizer.from_pretrained(v["id"], revision=v["revision"])
                  for k, v in MODELS.items()}
    path = hf_hub_download(SOURCE["id"], SOURCE["file"],
                           revision=SOURCE["revision"], repo_type="dataset")
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=256, columns=["id", "question", "context"]):
        rows.extend(batch.to_pylist())
        if len(rows) >= args.scan_rows:
            rows = rows[:args.scan_rows]
            break
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    samples, used_documents, used_texts = [], set(), set()
    for index in indices:
        row = rows[index]
        documents = {document_key(t) for t in row["context"]["title"]}
        if documents & used_documents:
            continue
        text, span = format_prompt(row)
        text_sha = sha(text.encode())
        if text_sha in used_texts:
            continue
        tokens = {key: encode(t, text) for key, t in tokenizers.items()}
        if any(not 64 <= len(t["ids"]) <= args.max_tokens for t in tokens.values()):
            continue  # No truncation: retain complete passages and question.
        correspondence = {
            "qwen_to_smol": align(tokens["qwen"]["offsets"], tokens["smol"]["offsets"]),
            "smol_to_qwen": align(tokens["smol"]["offsets"], tokens["qwen"]["offsets"])}
        probes = {key: [i for i, (a, b) in enumerate(t["offsets"])
                        if b > a and b > span[0] and a < span[1]]
                  for key, t in tokens.items()}
        if not all(probes.values()):
            raise ValueError("Question span has no token positions")
        samples.append({"id": str(row["id"]), "source_row": index,
                        "text": text, "text_sha256": text_sha,
                        "question_span": span, "question_positions": probes,
                        "documents": sorted(documents), "tokens": tokens,
                        "alignment": correspondence})
        used_documents.update(documents)
        used_texts.add(text_sha)
        if len(samples) == args.count:
            break
    if len(samples) != args.count:
        raise ValueError(f"Only {len(samples)} eligible prompts; increase --scan-rows or --max-tokens")
    validation = max(1, args.count // 5)
    for i, sample in enumerate(samples):
        sample["split"] = "validation" if i >= args.count - validation else "train"
    manifest = {"schema": 1, "source": SOURCE, "models": MODELS, "seed": args.seed,
                "max_tokens": args.max_tokens, "scan_rows": args.scan_rows,
                "prompt_format": "complete_hotpot_distractor_context_question_no_answer_v1",
                "feature_space": SPACE, "samples": samples}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "manifest.json", manifest)
    print(f"Prepared {len(samples)} prompts ({len(samples) - validation} train / {validation} validation)", flush=True)


def validate_manifest(manifest):
    if manifest["schema"] != 1 or manifest["source"] != SOURCE:
        raise ValueError("Manifest source/schema mismatch")
    if manifest["models"] != MODELS or manifest["feature_space"] != SPACE:
        raise ValueError("Manifest model revisions/feature space mismatch")
    seen_ids, seen_texts, seen_docs = set(), set(), set()
    for s in manifest["samples"]:
        if not s["id"].isalnum() or s["id"] in seen_ids:
            raise ValueError("Unsafe or duplicate sample ID")
        if sha(s["text"].encode()) != s["text_sha256"] or s["text_sha256"] in seen_texts:
            raise ValueError("Text hash mismatch or duplicate")
        if set(s["documents"]) & seen_docs:
            raise ValueError("Shared documents across selected samples")
        if s["split"] not in ("train", "validation"):
            raise ValueError("Unknown split")
        start, end = s["question_span"]
        if not 0 <= start < end <= len(s["text"]):
            raise ValueError("Invalid question span")
        for alias, tokens in s["tokens"].items():
            offsets = tokens["offsets"]
            if len(offsets) != len(tokens["ids"]) or not 64 <= len(offsets) <= manifest["max_tokens"]:
                raise ValueError("Invalid token count")
            if any(not 0 <= a <= b <= len(s["text"]) for a, b in offsets):
                raise ValueError("Invalid token offsets")
            positions = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
            if not positions or positions != s["question_positions"][alias]:
                raise ValueError("Invalid question token positions")
        for direction, (src, dst) in {
                "qwen_to_smol": ("qwen", "smol"), "smol_to_qwen": ("smol", "qwen")}.items():
            if s["alignment"][direction] != align(s["tokens"][src]["offsets"], s["tokens"][dst]["offsets"]):
                raise ValueError("Invalid alignment")
        seen_ids.add(s["id"])
        seen_texts.add(s["text_sha256"])
        seen_docs.update(s["documents"])


def geometry(config):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return {"layers": config.num_hidden_layers, "kv_heads": config.num_key_value_heads,
            "head_dim": dim, "width": dim * config.num_key_value_heads}


def validate_features(tensors, shape):
    names = {f"layer_{i}.{role}" for i in range(shape["layers"]) for role in ("k", "v")}
    if set(tensors) != names:
        raise ValueError("Missing/unexpected layers or roles")
    for name, value in tensors.items():
        if tuple(value.shape) != (shape["tokens"], shape["width"]) or value.dtype != torch.float32:
            raise ValueError(f"Wrong feature shape/dtype: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite features: {name}")


@torch.inference_mode()
def capture(model, ids, device):
    tensors, handles = {}, []
    def save(name):
        def hook(module, inputs, output):
            tensors[name] = output[0].detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        return hook
    try:
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, "k_norm"):
                raise ValueError("This pilot supports models without key normalization only")
            for role in ("k", "v"):
                projection = getattr(layer.self_attn, f"{role}_proj")
                handles.append(projection.register_forward_hook(save(f"layer_{i}.{role}")))
        model(input_ids=torch.tensor([ids], device=device),
              attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
              position_ids=torch.arange(len(ids), device=device)[None],
              use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    shape = dict(geometry(model.config), tokens=len(ids))
    validate_features(tensors, shape)
    return tensors, shape


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def extract(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    for alias in args.models:
        spec = MODELS[alias]
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        for s in manifest["samples"]:
            if encode(tokenizer, s["text"]) != s["tokens"][alias]:
                raise ValueError("Pinned tokenizer does not reproduce manifest")
        print(f"Loading {alias} on {device}, FP32", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"], revision=spec["revision"], dtype=torch.float32,
            attn_implementation="eager").eval().to(device)
        model.requires_grad_(False)
        if model.config.model_type != spec["type"]:
            raise ValueError("Unexpected model architecture")
        if geometry(model.config) != EXPECTED_GEOMETRY[alias]:
            raise ValueError("Unexpected pinned model geometry")
        destination = args.output / alias
        destination.mkdir(exist_ok=True)
        for s in manifest["samples"]:
            path = destination / f"{s['id']}.safetensors"
            if path.exists():
                raise ValueError(f"Refusing to overwrite existing features: {path}")
            identity = {"model": spec, "sample": s["id"], "text_sha256": s["text_sha256"],
                        "manifest_sha256": file_sha(manifest_path), "feature_space": SPACE}
            sync(device)
            start = time.perf_counter()
            tensors, shape = capture(model, s["tokens"][alias]["ids"], device)
            sync(device)
            elapsed = time.perf_counter() - start
            temporary = path.with_suffix(".tmp.safetensors")
            save_file(tensors, str(temporary), metadata={"identity": json.dumps(identity, sort_keys=True)})
            temporary.replace(path)
            restored = load_file(str(path))
            validate_features(restored, shape)
            if not all(torch.equal(value, restored[key]) for key, value in tensors.items()):
                raise ValueError("Serialization changed feature values")
            record = {"identity": identity, "shape": shape, "device": device,
                      "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                      "python": platform.python_version(), "extraction_seconds": elapsed,
                      "file_sha256": file_sha(path), "bytes": path.stat().st_size,
                      "payload_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}
            record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024)
            if device == "mps":
                record["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
            write_json(path.with_suffix(".json"), record)
            print(f"{alias}: {s['id']} {shape['tokens']} tokens, {elapsed:.2f}s, round-trip OK", flush=True)
            del tensors, restored
        del model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()


def verify(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    total_bytes, records = 0, []
    for alias, spec in MODELS.items():
        for sample in manifest["samples"]:
            path = args.output / alias / f"{sample['id']}.safetensors"
            record = json.loads(path.with_suffix(".json").read_text())
            identity = {"model": spec, "sample": sample["id"], "text_sha256": sample["text_sha256"],
                        "manifest_sha256": file_sha(manifest_path), "feature_space": SPACE}
            if record["identity"] != identity or file_sha(path) != record["file_sha256"]:
                raise ValueError("Shard provenance/hash mismatch")
            with safe_open(str(path), framework="pt") as archive:
                if json.loads(archive.metadata()["identity"]) != identity:
                    raise ValueError("Embedded provenance mismatch")
            tensors = load_file(str(path))
            if record["shape"] != dict(EXPECTED_GEOMETRY[alias], tokens=len(sample["tokens"][alias]["ids"])):
                raise ValueError("Shard geometry/token count mismatch")
            validate_features(tensors, record["shape"])
            total_bytes += path.stat().st_size
            records.append({"model": alias, "sample": sample["id"], **record["shape"],
                            "bytes": path.stat().st_size, "sha256": record["file_sha256"],
                            "extraction_seconds": record["extraction_seconds"]})
            del tensors
    report = {"validated": True, "prompts": len(manifest["samples"]),
              "splits": {s: sum(x["split"] == s for x in manifest["samples"])
                         for s in ("train", "validation")},
              "feature_space": SPACE, "models": MODELS, "source": SOURCE,
              "manifest_sha256": file_sha(manifest_path), "total_bytes": total_bytes,
              "shards": records,
              "alignment": [{"sample": s["id"], **{
                  d: {"exact": len(a["exact_pairs"]), "target_tokens": len(a["target_to_source"]),
                      "before_first": a["modes"].count("before_first")}
                  for d, a in s["alignment"].items()}} for s in manifest["samples"]]}
    write_json(args.output / "validation.json", report)
    print(f"Validated {len(records)} shards, {total_bytes / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "extract", "verify"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/paired-kv-pilot"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--scan-rows", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "verify": verify}[args.operation](args)
