"""Operations on embedding matrices and rankings: truncation, int8, Reciprocal Rank Fusion, the collapse diagnostic."""
import numpy as np

from .lexical import unit


def truncate(Z, k):
    """Keep the first k dimensions and rescale to length 1 (like the `dimensions` setting of embedding APIs)."""
    return unit(Z[:, :k])


def to_int8(Z):
    """Scalar quantization: one shared scale, every value rounded to an integer in [-127, 127]. Returns the values as floats."""
    scale = np.abs(Z).max() / 127
    return np.round(Z / scale).astype(np.int8).astype(np.float64) * scale


def mean_pairwise_cosine(Z):
    """Mean cosine over all pairs of distinct rows of a unit-length matrix. 1.0 means every vector is the same."""
    s = Z.sum(0)
    return float((s @ s - len(Z)) / (len(Z) * (len(Z) - 1)))


def ranks(scores, tiebreak):
    """Rank of every document for every query (1 = best). Equal scores are ordered by the fixed random `tiebreak` order."""
    perm = np.argsort(tiebreak)
    order = perm[np.argsort(-scores[:, perm], axis=1, kind="stable")]
    out = np.empty(scores.shape, dtype=np.int32)
    np.put_along_axis(out, order, np.arange(1, scores.shape[1] + 1, dtype=np.int32)[None, :], axis=1)
    return out


def rrf(rank_matrices, k=60):
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    return sum(1.0 / (k + r) for r in rank_matrices)
