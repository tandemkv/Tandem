"""Consistency with native Qwen, plus optional gold F1. Per-row metrics and paired summaries."""
import random
import re
import string
from collections import Counter
from statistics import mean

ARTICLES = {'the', 'a', 'an'}
# Capitalized only because they start the sentence.
OPENERS = {'he', 'she', 'it', 'they', 'his', 'her', 'their', 'this', 'that', 'there', 'in', 'on', 'at',
           'both', 'based', 'according'}
ENTITY = re.compile(r"\d[\d,.]*\d|\d|[A-Z][\w'&-]*(?:\s+(?:of|de|the|and|[A-Z][\w'&-]*))*")


def first_sentence(text):
    return re.split(r'(?<=[^A-Z0-9])\.(?:\s|$)|\n', text.strip(), maxsplit=1)[0]


def entities(text):
    """Numbers and capitalized spans in the first sentence, plus a leading yes/no."""
    sentence = first_sentence(text)
    found = set()
    for match in ENTITY.finditer(sentence):
        words = match.group().replace(',', '').split()
        while words and words[0].lower() in ARTICLES | OPENERS:
            words = words[1:]
        if words:
            found.add(' '.join(words).lower())
    lead = sentence.lower().split()[:1]
    if lead and lead[0].strip(string.punctuation) in ('yes', 'no'):
        found.add(lead[0].strip(string.punctuation))
    return found


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(w for w in text.split() if w not in ARTICLES)


def token_f1(prediction, reference):
    p, r = normalize(first_sentence(prediction)).split(), normalize(reference).split()
    common = sum((Counter(p) & Counter(r)).values())
    return 0.0 if not common else 2 * common / (len(p) + len(r))


def compare(native, candidate, native_first, candidate_first, native_nll, candidate_nll, gold=None):
    """Per-row metrics for one condition against native Qwen on the same case."""
    row = {
        'exact': candidate['tokens'] == native['tokens'],
        'top1': candidate_first.argmax().item() == native_first.argmax().item(),
        'kl': (native_first.exp() * (native_first - candidate_first)).sum().item(),
        'nll_gap': candidate_nll - native_nll,  # Extra loss on native's answer. 0 is a perfect copy.
        'entities': entities(candidate['answer']) == entities(native['answer']),
    }
    if gold is not None:
        row['gold_f1'] = token_f1(candidate['answer'], gold)
    return row


def bootstrap(values, samples=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def summarize(rows):
    """Mean and 95% bootstrap CI of every metric for every condition."""
    summary = {'cases': len(rows)}
    for name in rows[0]['conditions']:
        metrics = {}
        for key in rows[0]['conditions'][name]['metrics']:
            values = [float(r['conditions'][name]['metrics'][key]) for r in rows]
            metrics[key] = {'mean': mean(values), 'ci95': bootstrap(values)}
        summary[name] = metrics
    if rows[0].get('gold') is not None:
        summary['native_gold_f1'] = mean(token_f1(r['native']['answer'], r['gold']) for r in rows)
    return summary
