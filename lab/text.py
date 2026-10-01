"""Tokenizer (the auditor's, unchanged), word-overlap tiers and the trigram leak audit."""
import re

# Same stop words and token rule as the Document Risk Auditor's retrieval.py.
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or",
             "that", "the", "this", "to", "with", "will"}


def tokens(text):
    return [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]+", text.lower()) if t not in STOPWORDS and len(t) > 2]


def overlap(query, texts):
    """Share of the query's distinct content words that occur in the best-matching text of `texts`."""
    q = set(tokens(query))
    return max(len(q & set(tokens(t))) for t in texts) / len(q) if q else 0.0


# Tier cut points were fixed after looking at the overlap distribution of the queries and before any
# retrieval method was run on them (see docs/method.md, "Overlap tiers").
TIERS = (("low", 0.0, 0.5), ("mid", 0.5, 0.8), ("high", 0.8, 1.01))


def tier(ov):
    return next(name for name, lo, hi in TIERS if lo <= ov < hi)


def trigrams(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def containment(query, gram_set):
    """Share of the query's word trigrams that occur in a set of trigrams (1.0 means copied)."""
    g = trigrams(query)
    return len(g & gram_set) / len(g) if g else 0.0
