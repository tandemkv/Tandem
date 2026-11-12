"""Offline checks for checkpoint integrity and interrupted batch isolation."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/experiments"))
import cloud_prefill as cloud


class CloudTests(unittest.TestCase):
    def test_upload_checks_both_hash_formats_and_explicit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard, metadata, stale = [root / name for name in ("a.safetensors", "checkpoint.json", "stale")]
            shard.write_bytes(b"features")
            metadata.write_bytes(b"{}")
            stale.write_bytes(b"unfinished next batch")
            lfs = NS(rfilename="runs/test/a.safetensors", lfs=NS(sha256=cloud.base.file_sha(shard)))
            blob = NS(rfilename="runs/test/checkpoint.json", lfs=None,
                      blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest())
            class API:
                def upload_folder(self, **kwargs):
                    self.sent = kwargs["allow_patterns"]
                    return NS(oid="commit")

                def dataset_info(self, repo, **kwargs):
                    return NS(private=True, siblings=[lfs, blob])
            api = API()
            self.assertEqual(cloud.upload_verified(api, "repo", "runs/test", root, [shard, metadata]), "commit")
            self.assertEqual(api.sent, ["a.safetensors", "checkpoint.json"])
            self.assertTrue(stale.exists())
            lfs.lfs.sha256 = "wrong"
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                cloud.upload_verified(api, "repo", "runs/test", root, [shard, metadata])
            self.assertTrue(shard.exists())

    def test_resume_rejects_wrong_model_identity_shape_and_hash(self):
        sample = {"id": "abc", "text_sha256": "text",
                  "tokens": {alias: {"ids": [1, 2]} for alias in cloud.base.MODELS}}
        name = "qwen/abc.safetensors"
        record = {"sha256": "digest", "bytes": 20,
                  "shape": dict(cloud.base.EXPECTED_GEOMETRY["qwen"], tokens=2),
                  "identity": {"model": cloud.base.MODELS["qwen"], "sample": "abc",
                               "text_sha256": "text", "manifest_sha256": "manifest",
                               "feature_space": cloud.base.SPACE}}
        remote = {"runs/test/" + name: NS(lfs=NS(sha256="digest"), size=20)}
        def check(value):
            cloud.validate_receipts({name: value}, {"samples": [sample]}, "manifest", remote, "runs/test")
        check(record)
        for field, key, value in [("shape", "tokens", 3), ("identity", "sample", "other")]:
            changed = copy.deepcopy(record)
            changed[field][key] = value
            with self.assertRaises(ValueError):
                check(changed)
        remote["runs/test/" + name].lfs.sha256 = "changed"
        with self.assertRaises(ValueError):
            check(record)


if __name__ == "__main__":
    unittest.main()
