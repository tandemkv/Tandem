# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2",
#   "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Cluster-only prefill runner: bounded batches, verified upload, resumable receipts."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import paired_kv_dataset as base
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM


def remote_digest(entry, path):
    if entry.lfs:
        return entry.lfs.sha256 == base.file_sha(path)
    import hashlib
    data = path.read_bytes()
    return entry.blob_id == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def upload_verified(api, repo, prefix, folder, paths):
    # Explicit files keep interrupted/future batches out of this checkpoint.
    names = [p.relative_to(folder).as_posix() for p in paths]
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder,
                               path_in_repo=prefix, create_pr=False,
                               allow_patterns=names,
                               commit_message=f"Upload {prefix}: {len(paths)} files")
    info = api.dataset_info(repo, revision=commit.oid, files_metadata=True)
    if not info.private:
        raise RuntimeError("Dataset must remain private")
    entries = {s.rfilename: s for s in info.siblings}
    for path in paths:
        name = f"{prefix}/{path.relative_to(folder).as_posix()}"
        if name not in entries or not remote_digest(entries[name], path):
            raise RuntimeError(f"Remote hash mismatch: {name}; local batch retained")
    return commit.oid


def validate_receipts(receipts, manifest, manifest_sha, remote_names, prefix):
    expected = {f"{alias}/{s['id']}.safetensors": (alias, spec, s)
                for alias, spec in base.MODELS.items() for s in manifest["samples"]}
    if set(receipts) - expected.keys():
        raise ValueError("Unexpected shards in checkpoint")
    for name, record in receipts.items():
        alias, spec, sample = expected[name]
        identity = {"model": spec, "sample": sample["id"],
                    "text_sha256": sample["text_sha256"],
                    "manifest_sha256": manifest_sha, "feature_space": base.SPACE}
        shape = dict(base.EXPECTED_GEOMETRY[alias], tokens=len(sample["tokens"][alias]["ids"]))
        entry = remote_names.get(f"{prefix}/{name}")
        if (record["identity"] != identity or record["shape"] != shape
                or entry is None or entry.lfs is None
                or entry.lfs.sha256 != record["sha256"] or entry.size != record["bytes"]):
            raise ValueError("Remote resume receipt does not match uploaded shard")


def run(args):
    import fcntl
    if not torch.cuda.is_available() or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to your allocated GPU; CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated GPU")
    if args.batch < 1 or args.batch > 10 or not args.prefix.startswith("runs/") or ".." in args.prefix:
        raise ValueError("Use batch 1..10 and a unique runs/... prefix")
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / "runner.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    api = HfApi()
    info = api.dataset_info(args.repo, files_metadata=True)
    if not info.private:
        raise RuntimeError("Refusing a public dataset repository")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    if not (args.work / "manifest.json").exists():
        base.prepare(argparse.Namespace(output=args.work, count=args.count,
                     scan_rows=args.scan_rows, seed=args.seed, max_tokens=args.max_tokens))
    manifest = json.loads((args.work / "manifest.json").read_text())
    base.validate_manifest(manifest)
    if (len(manifest["samples"]) != args.count or manifest["seed"] != args.seed
            or manifest["max_tokens"] != args.max_tokens or manifest["scan_rows"] != args.scan_rows):
        raise ValueError("Work directory belongs to a different run")
    manifest_sha = base.file_sha(args.work / "manifest.json")
    receipts = {}
    remote_names = {s.rfilename: s for s in info.siblings}
    checkpoint_path = f"{args.prefix}/checkpoint.json"
    if checkpoint_path in remote_names:
        path = hf_hub_download(args.repo, checkpoint_path, repo_type="dataset", revision=info.sha)
        checkpoint = json.loads(Path(path).read_text())
        if checkpoint["manifest_sha256"] != manifest_sha:
            raise ValueError("Remote run prefix belongs to a different manifest")
        receipts = checkpoint["shards"]
        validate_receipts(receipts, manifest, manifest_sha, remote_names, args.prefix)
        print(f"Resuming {len(receipts)} verified shards", flush=True)
    elif any(name.startswith(args.prefix + "/") for name in remote_names):
        raise ValueError("Remote prefix exists without a checkpoint; inspect before continuing")
    batch_dir = args.work / "pending"
    batch_dir.mkdir(exist_ok=True)
    # A process may have died after uploading but before removing its batch.
    for path in batch_dir.rglob("*"):
        if path.is_file():
            entry = remote_names.get(f"{args.prefix}/{path.relative_to(batch_dir).as_posix()}")
            if entry is not None and remote_digest(entry, path):
                path.unlink()
    pending_names = []

    def flush(final=False):
        if final:
            expected = len(base.MODELS) * len(manifest["samples"])
            if len(receipts) != expected:
                raise ValueError("Not all paired shards are complete")
        (batch_dir / "manifest.json").write_bytes((args.work / "manifest.json").read_bytes())
        checkpoint = {"manifest_sha256": manifest_sha, "models": base.MODELS,
                      "feature_space": base.SPACE, "shards": receipts}
        base.write_json(batch_dir / "checkpoint.json", checkpoint)
        if final:
            base.write_json(batch_dir / "complete.json", {"complete": True,
                "manifest_sha256": manifest_sha, "shards": len(receipts),
                "prompts": len(manifest["samples"]), "models": base.MODELS,
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cuda": torch.version.cuda, "source": base.SOURCE})
        paths = [batch_dir / name for name in pending_names]
        paths += [batch_dir / "manifest.json", batch_dir / "checkpoint.json"]
        if final:
            paths.append(batch_dir / "complete.json")
        revision = upload_verified(api, args.repo, args.prefix, batch_dir, paths)
        print(f"Verified cloud checkpoint: {len(receipts)} shards at {revision}", flush=True)
        # Remove only files in our temporary batch, only after remote hash checks.
        for path in paths:
            path.unlink()
        pending_names.clear()
        base.write_json(args.work / "last_upload.json", {"repo": args.repo,
                        "prefix": args.prefix, "revision": revision, "complete": final})

    # A failed prior batch can be regenerated safely. Do not delete it before
    # upload: the new extraction atomically replaces its own shard/sidecar files.
    for alias, spec in base.MODELS.items():
        remaining = [s for s in manifest["samples"] if f"{alias}/{s['id']}.safetensors" not in receipts]
        if not remaining:
            continue
        tokenizer = base.AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        for sample in remaining:
            if base.encode(tokenizer, sample["text"]) != sample["tokens"][alias]:
                raise ValueError("Tokenizer no longer reproduces the manifest")
        model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
                   dtype=torch.float32, attn_implementation="eager").eval().to("cuda:0")
        model.requires_grad_(False)
        if base.geometry(model.config) != base.EXPECTED_GEOMETRY[alias]:
            raise ValueError("Unexpected model geometry")
        torch.cuda.reset_peak_memory_stats()
        for index, sample in enumerate(remaining, 1):
            name = f"{alias}/{sample['id']}.safetensors"
            path = batch_dir / name
            path.parent.mkdir(exist_ok=True)
            base.sync("cuda")
            start = time.perf_counter()
            tensors, shape = base.capture(model, sample["tokens"][alias]["ids"], "cuda:0")
            base.sync("cuda")
            identity = {"model": spec, "sample": sample["id"], "text_sha256": sample["text_sha256"],
                        "manifest_sha256": manifest_sha, "feature_space": base.SPACE}
            temporary = path.with_suffix(".tmp.safetensors")
            save_file(tensors, str(temporary), metadata={"identity": json.dumps(identity, sort_keys=True)})
            temporary.replace(path)
            restored = load_file(str(path))
            base.validate_features(restored, shape)
            if not all(torch.equal(value, restored[key]) for key, value in tensors.items()):
                raise ValueError("Serialization changed features")
            record = {"sha256": base.file_sha(path), "bytes": path.stat().st_size,
                      "shape": shape, "identity": identity,
                      "extraction_and_save_seconds": time.perf_counter() - start,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            base.write_json(path.with_suffix(".json"), record)
            receipts[name] = record
            pending_names.extend([name, str(Path(name).with_suffix(".json"))])
            del tensors, restored
            print(f"{alias} {index}/{len(remaining)}: {sample['id']} saved", flush=True)
            if index % args.batch == 0 or index == len(remaining):
                flush()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    flush(final=True)
    print("COMPLETE: all paired shards verified remotely", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="tandemkv/paired-kv-qwen-smollm")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--scan-rows", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--batch", type=int, default=5)
    run(parser.parse_args())
