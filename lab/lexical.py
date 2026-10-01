"""Lexical baselines on one text collection: TF-IDF as in the auditor, and BM25.

Every method reads the same token counts (text.tokens). Scores come back as a dense (queries x docs) array.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .text import tokens


def rowsum(M):
    return np.asarray(M.sum(1)).ravel()


class Index:
    def __init__(self, texts):
        counts = [Counter(tokens(t)) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for c in counts for w in c}))}
        self.N, V = len(texts), len(self.vocab)
        self.tf = self._matrix(counts)
        self.df = np.bincount(self.tf.indices, minlength=V)
        self.dl = rowsum(self.tf)  # document length in tokens

    def _matrix(self, counts):
        rows, cols, vals = [], [], []
        for r, c in enumerate(counts):
            for w, n in c.items():
                if w in self.vocab:  # words the index has never seen are dropped
                    rows.append(r), cols.append(self.vocab[w]), vals.append(n)
        return csr_matrix((vals, (rows, cols)), shape=(len(counts), len(self.vocab)), dtype=np.float64)

    def query_counts(self, queries):
        """Term counts of the queries over the index vocabulary, and the squared counts of unseen words."""
        counts = [Counter(tokens(q)) for q in queries]
        unseen = np.array([sum(n * n for w, n in c.items() if w not in self.vocab) for c in counts], dtype=float)
        return self._matrix(counts), unseen

    # --- TF-IDF exactly as retrieval.py: smoothed idf, raw counts, L2 normalization, cosine -------------
    def _tfidf_docs(self):
        idf = np.log((self.N + 1) / (self.df + 1)) + 1.0
        A = self.tf.multiply(idf).tocsr()
        norm = np.sqrt(rowsum(A.multiply(A)))
        return csr_matrix(A.multiply(1 / np.maximum(norm, 1e-300)[:, None])), idf

    def tfidf(self, queries):
        A, idf = self._tfidf_docs()
        Q, unseen = self.query_counts(queries)
        Qw = Q.multiply(idf).tocsr()
        # The auditor gives words missing from the index idf 1.0 and keeps them in the query norm.
        norm = np.sqrt(rowsum(Qw.multiply(Qw)) + unseen)
        return (Qw @ A.T).toarray() / np.maximum(norm, 1e-300)[:, None]

    # --- BM25, k1 and b fixed in advance and never tuned ---------------------------------------------------
    def bm25(self, queries, k1=1.2, b=0.75):
        idf = np.log(1 + (self.N - self.df + 0.5) / (self.df + 0.5))
        tf = self.tf.tocoo()
        norm = k1 * (1 - b + b * self.dl[tf.row] / self.dl.mean())
        W = csr_matrix((idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm), (tf.row, tf.col)), shape=self.tf.shape)
        return (self.query_counts(queries)[0] @ W.T).toarray()
