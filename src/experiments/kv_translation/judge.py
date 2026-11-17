"""LLM judge for whether a translated answer says the same thing as native Qwen's answer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
PROMPT = """Question: {question}

Answer A: {native}

Answer B: {candidate}

Do A and B give the same answer to the question? They must name the same entities, numbers, and dates. \
Wording, length, and extra explanation do not matter. Either answer may be cut off mid-sentence. \
Reply with exactly SAME or DIFFERENT."""


class Judge:
    def __init__(self, device='cuda'):
        name, revision = JUDGE
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.same, self.different = (self.tokenizer.encode(w, add_special_tokens=False)[0]
                                     for w in ('SAME', 'DIFFERENT'))

    @torch.inference_mode()
    def __call__(self, question, native, candidate):
        """True when the judge prefers SAME over DIFFERENT for the first reply token."""
        content = PROMPT.format(question=question, native=native, candidate=candidate)
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                 add_generation_prompt=True, return_tensors='pt')
        logits = self.model(input_ids=ids.to(self.model.device), logits_to_keep=1).logits[0, -1]
        return bool(logits[self.same] > logits[self.different])


def annotate(rows, judge):
    """Add the judge verdict and semantic agreement (judge SAME and matching entities) to each row."""
    for row in rows:
        for result in row['conditions'].values():
            metrics = result['metrics']
            metrics['judge'] = judge(row['question'], row['native']['answer'], result['answer'])
            metrics['agree'] = metrics['judge'] and metrics['entities']
    return rows
