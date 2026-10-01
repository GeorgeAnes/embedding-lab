"""A two-tower embedding model trained from scratch with a contrastive loss, in NumPy, with an analytic gradient.

THE MODEL, in one sentence: every word has a learned 64-number vector; a text is the mean of its words'
vectors, scaled to length 1.  Two towers means two separate word tables: tower A reads the short text (a
summary or a query), tower B reads the long text (the rest of the docstring).  This is the smallest encoder
that shows the idea.  A transformer replaces "mean of word vectors" with layers that let each word's vector
depend on the words around it (and then pools the word vectors into one); the contrastive loss keeps the same form.

THE DATA: pairs (a_i, b_i) = (summary line, rest of the same docstring).  Matching pairs should score high,
every other combination in the batch should score low.

THE LOSS.  Let x be one text as a row of mean-pooling weights over the vocabulary (1/n for each of its n words),
W the word table (one 64-number row per word), and
    u = x W,   z = u / |u|                          (|u| is the length of u; z has length 1, so z_a . z_b is the cosine)
For a batch of B pairs let Za and Zb be the B x 64 matrices whose rows are z_a[i] and z_b[j], and
    S = Za Zb^T / tau,   so   S[i,j] = z_a[i] . z_b[j] / tau        (tau is the temperature)
Symmetric InfoNCE:
    L = 1/2 * ( -1/B sum_i log softmax_j(S[i,:])[i]      "which b belongs to a_i?"
              + -1/B sum_j log softmax_i(S[:,j])[j] )    "which a belongs to b_j?"
The B-1 other pairs in the batch are the negatives.  Without them the loss could be pushed to its minimum
by mapping every text to the same vector, and training does exactly that (see the ablation in docs/results.md).
The ablation (negatives=False) replaces L by -1/B sum_i S[i,i]: it only pulls matching pairs together.

THE GRADIENT.  Write P = row-wise softmax of S, Q = column-wise softmax of S, and I = the B x B identity matrix.
For a softmax cross-entropy the derivative with respect to the scores is (probabilities - one-hot), so
    dL/dS = ( (P - I) + (Q - I) ) / (2B)
Since S = Za Zb^T / tau:            dL/dZa = dL/dS  Zb / tau,      dL/dZb = (dL/dS)^T Za / tau
Normalizing z = u/|u| has Jacobian (I - z z^T)/|u|, so   dL/du = ( dL/dz - z (z . dL/dz) ) / |u|   (row by row)
Finally u = x W, so                 dL/dW = X^T dL/du       (X = the B x V matrix with one row x per text)
tests/test_contrastive.py checks all of this against finite differences.
"""
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix

from .text import tokens


# ---- 1. Text to numbers -----------------------------------------------------------------------------------
def build_vocab(texts, min_df=2):
    """Words that occur in at least `min_df` documents, sorted, as {word: column}."""
    df = Counter(w for t in texts for w in set(tokens(t)))
    return {w: i for i, w in enumerate(sorted(w for w, n in df.items() if n >= min_df))}


def to_ids(text, vocab):
    return np.array([vocab[w] for w in tokens(text) if w in vocab], dtype=np.int64)


def bag(id_lists, V, drop=0.0, rng=None):
    """Sparse matrix of mean-pooling weights: row r has 1/len at each word id of text r (dropout removes words)."""
    rows, cols, vals = [], [], []
    for r, ids in enumerate(id_lists):
        if drop and len(ids) > 1:
            ids = ids[rng.random(len(ids)) >= drop]  # a text can lose every word; then its row stays zero
        rows.append(np.full(len(ids), r)), cols.append(ids), vals.append(np.full(len(ids), 1.0 / max(len(ids), 1)))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(id_lists), V)).tocsr()


# ---- 2. Encoder -------------------------------------------------------------------------------------------
def encode(X, W):
    """u = x W (mean of word vectors), its length, and z = u / |u|."""
    u = X @ W
    n = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    return u, n, u / n


# ---- 3. Loss and its gradient -----------------------------------------------------------------------------
def log_softmax(S, axis):
    m = S.max(axis=axis, keepdims=True)
    return S - m - np.log(np.exp(S - m).sum(axis=axis, keepdims=True))


def loss_and_grads(XA, XB, WA, WB, tau, negatives=True):
    """Symmetric InfoNCE loss on one batch and its gradients with respect to the two word tables."""
    _, nA, ZA = encode(XA, WA)
    _, nB, ZB = encode(XB, WB)
    B = len(ZA)
    S = ZA @ ZB.T / tau
    if negatives:
        logP, logQ = log_softmax(S, 1), log_softmax(S, 0)
        loss = -(np.diag(logP).mean() + np.diag(logQ).mean()) / 2
        dS = (np.exp(logP) + np.exp(logQ) - 2 * np.eye(B)) / (2 * B)
    else:  # ablation: only pull matching pairs together, nothing pushes anything apart
        loss = -np.diag(S).mean()
        dS = -np.eye(B) / B
    dZA, dZB = dS @ ZB / tau, dS.T @ ZA / tau
    dUA = (dZA - ZA * (ZA * dZA).sum(1, keepdims=True)) / nA
    dUB = (dZB - ZB * (ZB * dZB).sum(1, keepdims=True)) / nB
    return loss, XA.T @ dUA, XB.T @ dUB


# ---- 4. Training loop with Adam ---------------------------------------------------------------------------
def train(a_ids, b_ids, V, d=64, tau=0.1, steps=800, batch=256, lr=0.01, drop=0.2, seed=0, negatives=True, every=100):
    """Learn the two word tables.  a_ids[i], b_ids[i] are the word ids of pair i.  Returns WA, WB, loss history."""
    rng = np.random.default_rng(seed)
    W = [rng.normal(0, 0.1, (V, d)) for _ in range(2)]
    m, v = [np.zeros_like(w) for w in W], [np.zeros_like(w) for w in W]  # Adam moment estimates
    order, pos, history, window = rng.permutation(len(a_ids)), 0, [], []
    for step in range(1, steps + 1):
        if pos + batch > len(order):  # next epoch: reshuffle the pairs
            order, pos = rng.permutation(len(a_ids)), 0
        idx, pos = order[pos:pos + batch], pos + batch
        XA = bag([a_ids[i] for i in idx], V, drop, rng)
        XB = bag([b_ids[i] for i in idx], V, drop, rng)
        loss, *grads = loss_and_grads(XA, XB, W[0], W[1], tau, negatives)
        for k in range(2):  # Adam: running mean and running mean of squares of the gradient, bias-corrected
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            W[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        window.append(loss)
        if step % every == 0 or step == 1:
            history.append((step, float(np.mean(window))))
            window = []
    return W[0], W[1], history


# ---- 5. Using the trained model ---------------------------------------------------------------------------
class Model:
    def __init__(self, vocab, WA, WB):
        self.vocab, self.WA, self.WB = vocab, WA, WB

    def _encode(self, W, texts, chunk=512):
        """Encode texts in chunks (a dense bag matrix for all texts at once would be large). Returns (u, z)."""
        out = [encode(bag([to_ids(t, self.vocab) for t in texts[i:i + chunk]], len(self.vocab)), W) for i in range(0, len(texts), chunk)]
        return np.vstack([o[0] for o in out]), np.vstack([o[2] for o in out])

    def embed_queries(self, texts):
        """Tower A. Returns (u, z): the vector before and after scaling to length 1."""
        return self._encode(self.WA, texts)

    def embed_docs(self, summaries, rests):
        """A document is the sum of its two halves (tower A on the summary, tower B on the rest), scaled to length 1."""
        u = self._encode(self.WA, summaries)[1] + self._encode(self.WB, rests)[1]
        return u, u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
