"""Check that reused reports compare the same models, prompts and native tokens."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import paired_kv_dataset as base
from finish_continuation import check_report, compare_reports


class FinishTests(unittest.TestCase):
    def test_report_reuse_requires_matching_checkpoint_and_generation_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            path.write_bytes(b'checkpoint identity fixture')
            report = {'checkpoint_sha256': base.file_sha(path), 'generation': {'mode': 'greedy', 'max_new_tokens': 32},
                      'split': 'validation', 'samples': [{'id': str(i)} for i in range(40)]}
            check_report(report, path)
            report['handoff'] = 'question'
            with self.assertRaises(ValueError):
                check_report(report, path)
            report['handoff'] = 'last-token'
            path.write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                check_report(report, path)
            report['checkpoint_sha256'] = base.file_sha(path)
            report['generation']['max_new_tokens'] = 1
            with self.assertRaises(ValueError):
                check_report(report, path)

    def test_comparison_rejects_different_native_continuations(self):
        report = {'dataset_revision': 'same', 'samples': [{'id': 'a', 'native_tokens': [3, 4]}],
                  'summary': {'methods': {'attention': {'first_token_agreement': .5}}}}
        reports = {1: report, 32: copy.deepcopy(report)}
        self.assertEqual(set(compare_reports(reports)), {'1', '32'})
        reports[32]['samples'][0]['native_tokens'] = [3, 5]
        with self.assertRaisesRegex(ValueError, 'continuations differ'):
            compare_reports(reports)


if __name__ == '__main__':
    unittest.main()
