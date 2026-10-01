"""Copy of the retrieval functions of the Enterprise AI Document Risk Auditor (backend/app/core/retrieval.py), with the
type hints removed and one loop variable renamed in _tokens; the logic is unchanged. Used only to check that
lab.lexical.Index.tfidf gives the same scores."""
import math
import re
from collections import Counter

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "will"}


def _tokens(text):
    return [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]+", text.lower()) if t not in STOPWORDS and len(t) > 2]


def _build_idf(texts):
    documents = [_tokens(text) for text in texts]
    total = max(len(documents), 1)
    document_frequency = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    return {token: math.log((total + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}


def _tfidf_vector(text, idf):
    counts = Counter(_tokens(text))
    if not counts:
        return {}
    vector = {token: count * idf.get(token, 1.0) for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {token: value / norm for token, value in vector.items()}


def _cosine(left, right):
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) < len(right) else (right, left)
    return sum(value * larger.get(token, 0.0) for token, value in smaller.items())
