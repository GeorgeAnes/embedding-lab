import numpy as np

from lab import contrastive as C
from lab.vecops import mean_pairwise_cosine


def small_problem(seed=3, V=30, d=5, B=6):
    rng = np.random.default_rng(seed)
    def bag():
        X = rng.random((B, V)) * (rng.random((B, V)) < 0.3)
        return X / np.maximum(X.sum(1, keepdims=True), 1e-9)
    return bag(), bag(), rng.normal(0, 0.3, (V, d)), rng.normal(0, 0.3, (V, d))


def test_gradient_matches_finite_differences():
    """The analytic gradient of the loss agrees with central differences for every entry of both word tables, with and without negatives."""
    for negatives in (True, False):
        XA, XB, WA, WB = small_problem()
        _, gA, gB = C.loss_and_grads(XA, XB, WA, WB, 0.2, negatives)
        eps = 1e-6
        for W, g in ((WA, gA), (WB, gB)):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    old = W[i, j]
                    W[i, j] = old + eps
                    up = C.loss_and_grads(XA, XB, WA, WB, 0.2, negatives)[0]
                    W[i, j] = old - eps
                    down = C.loss_and_grads(XA, XB, WA, WB, 0.2, negatives)[0]
                    W[i, j] = old
                    numeric = (up - down) / (2 * eps)
                    assert abs(numeric - g[i, j]) <= 1e-6 * max(1.0, abs(numeric)), (negatives, i, j, numeric, g[i, j])


def toy_pairs(n=64, pool=20, seed=0):
    """Pair i: one word of its own plus five from a shared pool of 20, on each side. The two sides use different words."""
    rng = np.random.default_rng(seed)
    off = n + pool
    a = [np.concatenate([[i], n + rng.integers(0, pool, 5)]) for i in range(n)]
    b = [np.concatenate([[i + off], off + n + rng.integers(0, pool, 5)]) for i in range(n)]
    return a, b, 2 * off


def train_toy(**kw):
    a, b, V = toy_pairs()
    WA, WB, hist = C.train(a, b, V, steps=300, batch=32, drop=0.0, **kw)
    Z = C.encode(C.bag(a, V), WA)[2]
    return hist, mean_pairwise_cosine(Z)


def test_training_lowers_the_loss_and_is_deterministic():
    hist, _ = train_toy(seed=1, every=50)
    assert hist[-1][1] < hist[0][1] / 3
    assert train_toy(seed=1, every=50)[0] == hist


def test_without_negatives_all_vectors_collapse():
    """The ablation reported in docs/results.md: pulling pairs together with nothing pushing apart makes every vector the same."""
    assert train_toy()[1] < 0.3
    assert train_toy(negatives=False)[1] > 0.9
