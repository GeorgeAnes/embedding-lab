import math

import numpy as np

from lab.lexical import Index
from . import auditor_reference as ref

TEXTS = ["Fit a curve to data with least squares", "Solve a differential equation numerically", "Least squares solution of a linear system of equations",
         "Random sampling of a normal distribution", "Curve fitting of data with noise, data data"]
QUERIES = ["fit curve to noisy data", "least squares equations", "xylophone zebra", "differential equation data data"]


def test_tfidf_matches_the_auditors_implementation():
    idf = ref._build_idf(TEXTS)
    docs = [ref._tfidf_vector(t, idf) for t in TEXTS]
    expected = np.array([[ref._cosine(ref._tfidf_vector(q, idf), d) for d in docs] for q in QUERIES])
    np.testing.assert_allclose(Index(TEXTS).tfidf(QUERIES), expected, atol=1e-9)


def test_bm25_matches_the_formula():
    texts = ["alpha beta", "alpha alpha gamma", "delta"]
    N, df, avg = 3, 2, 2.0  # token counts 2, 3, 1
    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
    def score(f, length):
        return idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / avg))
    got = Index(texts).bm25(["alpha"])[0]
    np.testing.assert_allclose(got, [score(1, 2), score(2, 3), 0.0])
