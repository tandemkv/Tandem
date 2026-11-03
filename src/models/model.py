from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:

    def __init__(self,*, model_name, kv_cache=None):
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cache = None
        self._initial_cache = kv_cache

    def run(self, prompt):
        """Generate an answer, using the supplied cache on the first run only.

        With an initial cache, prompt must contain only the uncached suffix in
        this model's tokenization. The supplied cache is extended in place.
        Later runs process their prompts from scratch.
        """
        generation_kwargs = {}
        if self._initial_cache is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            past_length = self._initial_cache.get_seq_length()
            mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [mask.new_ones((mask.shape[0], past_length)), mask], dim=1
            )
            generation_kwargs["past_key_values"] = self._initial_cache
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                **generation_kwargs,
            )

        self._initial_cache = None
        new_tokens = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        self.cache = outputs.past_key_values

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_kv_cache(self):
        return self.cache
