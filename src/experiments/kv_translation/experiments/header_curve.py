"""Bridge QA on output3: does the translated header explain the gap between 75% native body and full native?

The copy oracle so far kept the header line ("Passage 1:") translated, while full native has it native.
Body 100% native with a translated header, 75% and 100% with a native header, and contiguous native prefixes
(header plus lead third, or header plus the first sentence) with the tail translated or masked out.
Also output3 with layer 0 replaced by Qwen's own, which depends only on the token IDs (deployable).
Cases and the `passage1_needed` subset match `bridge`.
"""
from experiments.bridge import SUBSETS, cases  # noqa: F401

CONDITIONS = ['question_only', 'output3', 'oracle_scattered100_output3', 'oracle_hscattered75_output3',
              'oracle_hscattered100_output3', 'oracle_hlead33_output3', 'masked_hlead33_output3',
              'oracle_hsentence_output3', 'masked_hsentence_output3', 'exact0_output3']
