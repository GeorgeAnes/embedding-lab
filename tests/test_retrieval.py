import math

import numpy as np

from lab.evaluate import summarize
from lab.lexical import Index
from lab.vecops import mean_pairwise_cosine, ranks, rrf, to_int8, truncate
from . import auditor_reference as ref

TEXTS = ["Fit a curve to data with least squares", "Solve a differential equation numerically", "Least squares solution of a linear system of equations",
         "Random sampling of a normal distribution", "Curve fitting of data with noise, data data"]
TWO_TOPICS = ["apple banana cherry", "banana cherry date", "apple cherry date", "engine wheel brake", "wheel brake tire", "engine brake tire"]
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


def test_lsa_document_similarities_match_a_truncated_svd():
    ix = Index(TWO_TOPICS)
    A = ix._tfidf_docs()[0].toarray()
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    D = U[:, :2] * S[:2]
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Z = ix.lsa_docs(k=2)[1]
    np.testing.assert_allclose(Z @ Z.T, D @ D.T, atol=1e-8)


def test_ties_are_broken_by_the_fixed_random_order_not_by_position():
    tiebreak = np.random.default_rng(0).permutation(50)
    r = ranks(np.zeros((1, 50)), tiebreak)[0]
    assert sorted(r) == list(range(1, 51)) and list(r) != list(range(1, 51))
    assert list(ranks(np.array([[0.1, 0.9, 0.5]]), np.arange(3))[0]) == [3, 1, 2]


def test_rrf_uses_one_over_k_plus_rank():
    a, b = np.array([[1, 2, 3]]), np.array([[3, 1, 2]])
    np.testing.assert_allclose(rrf([a, b], k=60)[0], [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])


def test_vector_operations():
    Z = np.random.default_rng(0).normal(size=(20, 8))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(truncate(Z, 3), axis=1), 1)
    assert np.abs(to_int8(Z) - Z).max() <= np.abs(Z).max() / 127 / 2 + 1e-12
    same = np.tile(Z[:1], (5, 1))
    assert abs(mean_pairwise_cosine(same) - 1) < 1e-12


def test_summarize_metrics_and_paired_difference():
    best = {"a": np.array([1, 2, 20, 5]), "b": np.array([1, 2, 20, 5]), "c": np.array([3, 3, 3, 3])}
    out = summarize(best, np.array([0, 0, 1, 1]), "a", {"all": np.ones(4, bool)}, B=200)
    assert out["a"]["all"]["r10"][0] == 0.75
    assert abs(out["a"]["all"]["mrr"][0] - (1 + 0.5 + 0 + 0.2) / 4) < 1e-12
    assert out["b"]["all"]["d_r10"] == [0.0, 0.0, 0.0]
    assert out["c"]["all"]["r10"][0] == 1.0 and out["c"]["all"]["d_r10"][0] == 0.25
