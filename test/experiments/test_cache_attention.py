"""Offline architecture, causal alignment, learning and differentiable replay tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
import paired_kv_dataset as base
from cache_attention import CacheAttention
from train_cache_attention import causal_positions, stack_features, receiver_logits, logit_kl, fit_statistics_and_ridge
from types import SimpleNamespace


def config():
    return dict(source_layers=3, source_heads=2, target_layers=2, target_heads=1,
                head_dim=4, width=16, attention_heads=2, source_id=1, target_id=0)


class MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_residual_and_no_cross_position_leakage(self):
        torch.manual_seed(2)
        mapper = CacheAttention(config())
        mapper.ridge.normal_()
        x = torch.randn(5, 2, 6, 4)
        torch.testing.assert_close(mapper(x), mapper(x, affine_only=True))
        torch.nn.init.normal_(mapper.output.weight)
        first = mapper(x)
        x[2] += 100
        changed = mapper(x)
        torch.testing.assert_close(first[:2], changed[:2])
        torch.testing.assert_close(first[3:], changed[3:])
        self.assertFalse(torch.allclose(first[2], changed[2]))
        self.assertEqual(first.shape, (5, 2, 2, 4))

    def test_role_separation(self):
        mapper = CacheAttention(config())
        torch.nn.init.normal_(mapper.output.weight)
        x = torch.randn(3, 2, 6, 4)
        before = mapper(x)
        x[:, 0] *= 100
        torch.testing.assert_close(before[:, 1], mapper(x)[:, 1])

    def test_can_learn_small_nonlinear_training_problem(self):
        torch.manual_seed(3)
        mapper = CacheAttention(config())
        x = torch.randn(8, 2, 6, 4)
        y = torch.tanh(x[:, :, :2] + x[:, :, 4:6])
        optimizer = torch.optim.Adam(mapper.parameters(), lr=.01)
        initial = (mapper(x) - y).square().mean().item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = (mapper(x) - y).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess((mapper(x) - y).square().mean().item(), initial * .15)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in mapper.parameters()))

    def test_before_first_alignment_is_excluded(self):
        sample = {'alignment': {'smol_to_qwen': {'target_to_source': [0, 0, 1]}},
                  'tokens': {'smol': {'offsets': [[0, 3], [3, 5]]},
                             'qwen': {'offsets': [[0, 1], [1, 3], [3, 5]]}}}
        alignment, mask = causal_positions(sample, 'smol', 'qwen')
        self.assertEqual(alignment.tolist(), [0, 0, 1])
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_train_only_statistics_and_affine_recovery(self):
        torch.manual_seed(5)
        x = torch.randn(96, 2, 6, 4)
        # Target layers use the proportional source layers (0 and 2).
        y = torch.stack([x[:, :, :2].sum(2), x[:, :, 4:6].sum(2)], dim=2) * 2 + 3
        class Data:
            train = ['training']
            validation = ['held_out']
            def pair(self, sample, stride):
                if sample != 'training':
                    raise AssertionError('Validation must not influence fitting statistics')
                return x, y
        mapper = CacheAttention(config())
        fit_statistics_and_ridge(mapper, Data(), SimpleNamespace(device='cpu', stride=1, ridge=1e-5))
        self.assertLess((mapper.predict(x, True) - y).square().mean().item(), 1e-5)

    def test_native_replay_and_gradient_for_both_architectures(self):
        for config_type, model_type in [(Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)]:
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(7)
                cfg = config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
                cfg._attn_implementation = 'eager'
                receiver = model_type(cfg).eval()
                receiver.requires_grad_(False)
                ids = torch.tensor([[2, 4, 7, 11, 6, 9]])
                tensors, shape = base.capture(receiver, ids[0].tolist(), 'cpu')
                features = stack_features(tensors, shape)
                with torch.no_grad():
                    teacher = receiver(input_ids=ids, use_cache=False, logits_to_keep=2).logits
                    control = receiver_logits(receiver, features[:-2], ids[:, -2:])
                torch.testing.assert_close(control, teacher, rtol=1e-4, atol=1e-5)
                self.assertLess(abs(logit_kl(control, teacher).item()), 1e-5)
                perturbed = (features[:-2] + torch.randn_like(features[:-2]) * .2).requires_grad_()
                student = receiver_logits(receiver, perturbed, ids[:, -2:])
                loss = logit_kl(student, teacher)
                loss.backward()
                self.assertGreater(perturbed.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(perturbed.grad).all())
                self.assertTrue(all(p.grad is None for p in receiver.parameters()))


if __name__ == '__main__':
    unittest.main()
