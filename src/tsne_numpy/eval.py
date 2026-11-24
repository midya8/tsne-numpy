"""How much neighborhood structure survives the embedding."""

import numpy as np

from .affinity import sq_dists


def knn_overlap(D2_orig, D2_emb, k=10):
    """Fraction of the k nearest neighbors in the input that are still
    k nearest neighbors in the embedding, averaged over points."""
    n = D2_orig.shape[0]
    rng = np.arange(n)
    no = np.argsort(D2_orig, axis=1)[:, 1:k + 1]
    ne = np.argsort(D2_emb, axis=1)[:, 1:k + 1]
    hits = 0
    for i in range(n):
        hits += len(set(no[i]) & set(ne[i]))
    return hits / (n * k)


def trustworthiness(X, Y, k=10):
    """1 minus the penalty for points that look close in the embedding
    but are far in the input (the sklearn definition)."""
    n = X.shape[0]
    Do, De = sq_dists(X), sq_dists(Y)
    ro = np.argsort(Do, axis=1)
    re = np.argsort(De, axis=1)[:, 1:k + 1]
    # rank of each embedded neighbor within the input ordering
    rank = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        rank[i, ro[i]] = np.arange(n)
    pen = sum(max(0.0, float(rank[i, j]) - k) for i in range(n) for j in re[i])
    return 1.0 - (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))) * pen
