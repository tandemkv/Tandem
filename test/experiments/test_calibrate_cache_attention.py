"""Continuation alignment and gradients, without model or dataset downloads."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from requests import Response
from requests.exceptions import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import paired_kv_dataset as base
from train_cache_attention import stack_features, receiver_logits, logit_kl
from calibrate_cache_attention import (CheckpointPublisher, continuation_inputs, teacher_logits, stopping_update,
                                       save_progress, restore_progress)


class ContinuationTests(unittest.TestCase):
    def test_publication_throttles_commits_and_preserves_rate_limit_across_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(work=Path(tmp))
            calls, clock = [], [0.]
            publisher = CheckpointPublisher(args, upload=lambda _: calls.append(1), clock=lambda: clock[0])
            self.assertTrue(publisher())
            clock[0] = 10.
            self.assertFalse(publisher())
            self.assertEqual(len(calls), 1)
            self.assertTrue(publisher(final=True))
            response = Response()
            response.status_code = 429
            response.headers['Retry-After'] = '7200'
            def rate_limit(_):
                raise HTTPError('rate limited', response=response)
            publisher.upload = rate_limit
            clock[0] = 700.
            self.assertFalse(publisher())
            self.assertTrue(publisher.pending.exists())
            resumed = CheckpointPublisher(args, upload=lambda _: calls.append(1), clock=lambda: clock[0])
            self.assertFalse(resumed(final=True))
            self.assertEqual(len(calls), 2)
            clock[0] = 7901.
            self.assertTrue(resumed(final=True))
            self.assertFalse(resumed.pending.exists())

    def test_publication_does_not_hide_authentication_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = Response()
            response.status_code = 401
            def fail(_):
                raise HTTPError('unauthorized', response=response)
            publisher = CheckpointPublisher(SimpleNamespace(work=Path(tmp)), upload=fail)
            with self.assertRaises(HTTPError):
                publisher()

    def test_stopping_requires_five_unimproved_passes_and_resets_on_improvement(self):
        best, stale = 2., 0
        for _ in range(4):
            best, stale, improved = stopping_update(2., best, stale)
            self.assertFalse(improved)
            self.assertLess(stale, 5)
        best, stale, improved = stopping_update(1.9, best, stale)
        self.assertTrue(improved)
        self.assertEqual(stale, 0)
        for _ in range(5):
            best, stale, _ = stopping_update(2., best, stale)
        self.assertEqual(stale, 5)
        with self.assertRaises(ValueError):
            stopping_update(float('nan'), best, stale)

    def test_resume_preserves_adam_updates_rng_and_partial_pass(self):
        torch.manual_seed(21)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        provenance = dict.fromkeys(('parent_revision', 'parent_checkpoint', 'parent_sha256',
            'parent_run', 'tokens', 'handoff_prompt_tokens', 'lr', 'seed', 'train_ids',
            'validation_ids', 'pilot', 'objective'), 'same')
        run = {'parent': 'test'}
        state = dict(epoch=1, position=1, steps=4, best=1.5, stale=2,
                     epoch_loss_sum=2., epoch_loss_count=1)
        x = torch.tensor([[1., 2.]])
        def step(m, opt):
            opt.zero_grad()
            m(x).square().sum().backward()
            opt.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_progress(out, model, optimizer, run, provenance, state)
            expected_random = torch.rand(3)
            step(model, optimizer)
            other = torch.nn.Linear(2, 1)
            other_opt = torch.optim.AdamW(other.parameters(), lr=.01)
            restored, reset = restore_progress(out, other, other_opt, run, provenance, 3, 'cpu')
            self.assertFalse(reset)
            self.assertEqual(restored, state)
            torch.testing.assert_close(torch.rand(3), expected_random)
            step(other, other_opt)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                restore_progress(out, other, other_opt, run, dict(provenance, tokens='changed'), 3, 'cpu')

    def test_legacy_resume_keeps_completed_passes_and_reports_optimizer_reset(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        provenance = dict.fromkeys(('parent_revision', 'parent_checkpoint', 'parent_sha256',
            'parent_run', 'tokens', 'handoff_prompt_tokens', 'lr', 'seed', 'train_ids',
            'validation_ids', 'pilot', 'objective'), 'same')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            torch.save({'model': model.state_dict(), 'run': {}, 'calibration': provenance,
                        'steps': 5}, out / 'last.pt')
            (out / 'before.json').write_text(json.dumps({'summary': {'kl': 2.}}))
            rows = [dict(steps=2, prompts_seen_this_epoch=2, validation={'kl': 1.5}),
                    dict(steps=4, prompts_seen_this_epoch=2, validation={'kl': 1.6})]
            (out / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            state, reset = restore_progress(out, model, optimizer, {}, provenance, 2, 'cpu')
            self.assertTrue(reset)
            self.assertEqual((state['epoch'], state['position'], state['steps']), (2, 1, 5))
            self.assertEqual((state['best'], state['stale']), (1.5, 1))

    def test_continuation_alignment_including_one_token_and_eos(self):
        prompt = torch.tensor([[3, 5, 7]])
        for continuation in ([9], [9, 11, 2]):
            full, suffix = continuation_inputs(prompt, continuation)
            self.assertEqual(full.tolist(), [[3, 5, 7] + list(continuation[:-1])])
            self.assertEqual(suffix.tolist(), [[7] + list(continuation[:-1])])
            self.assertEqual(suffix.shape[1], len(continuation))
        with self.assertRaises(ValueError):
            continuation_inputs(prompt, [])

    def test_native_continuation_control_and_frozen_receiver_gradients(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        config._attn_implementation = 'eager'
        receiver = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        prompt = torch.tensor([[3, 5, 7, 9, 2, 1]])
        tensors, geometry = base.capture(receiver, prompt[0].tolist(), 'cpu')
        features = stack_features(tensors, geometry)[:-1]
        continuation = [4, 6, 8, 2]
        teacher = teacher_logits(receiver, prompt, continuation)
        _, suffix = continuation_inputs(prompt, continuation)
        control = receiver_logits(receiver, features, suffix)
        torch.testing.assert_close(control, teacher, atol=1e-5, rtol=1e-4)
        self.assertFalse(teacher.requires_grad)
        translated = (features + .15 * torch.randn_like(features)).requires_grad_()
        loss = logit_kl(receiver_logits(receiver, translated, suffix), teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(translated.grad).all())
        self.assertGreater(translated.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in receiver.parameters()))
        # Later continuation positions must also teach the translated prefix.
        translated.grad = None
        logit_kl(receiver_logits(receiver, translated, suffix)[:, 1:], teacher[:, 1:]).backward()
        self.assertGreater(translated.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
