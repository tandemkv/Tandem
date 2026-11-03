"""Real Qwen cache handoff test; first run downloads the model weights."""
import unittest

from src.models.model import Model


class ModelCacheTests(unittest.TestCase):
    def test_cache_handoff_to_fresh_qwen(self):
        model_name = 'Qwen/Qwen2.5-0.5B'
        prompt = 'The capital of France is'

        first_model = Model(model_name=model_name)
        first_answer = first_model.run(prompt)
        cache = first_model.get_kv_cache()
        self.assertIsNotNone(cache)
        original_cache_length = cache.get_seq_length()
        self.assertGreater(original_cache_length, 0)
        print('First answer:', first_answer)
        del first_model

        second_model = Model(model_name=model_name, kv_cache=cache)
        second_answer = second_model.run(prompt)
        print('Answer with imported cache:', second_answer)

        self.assertTrue(first_answer.strip())
        self.assertTrue(second_answer.strip())
        self.assertIs(second_model.get_kv_cache(), cache)
        self.assertGreater(cache.get_seq_length(), original_cache_length)


if __name__ == '__main__':
    unittest.main()
