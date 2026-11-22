"""Key/value swap: which of the translated keys or values loses the most against native Qwen."""
from data import hotpot_test

CONDITIONS = ['question_only', 'distilled', 'native_keys_distilled', 'native_values_distilled']


def cases():
    return hotpot_test()
