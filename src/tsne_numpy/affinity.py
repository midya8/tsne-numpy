"""Pairwise distances and the Gaussian affinities t-SNE starts from."""

import numpy as np


def sq_dists(X):
    d2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :]
    d2 -= 2.0 * X @ X.T
    np.fill_diagonal(d2, 0.0)
    return np.maximum(d2, 0.0)


def row_conditional(d2_i, beta, self_idx):
    """p_j|i with beta = 1/(2 sigma_i^2). The self pair is masked out."""
    p = np.exp(-d2_i * beta)
    p[self_idx] = 0.0
    return p


def target_perplexity_row(d2_i, perplexity, self_idx, tol=1e-5, max_iter=60):
    """Binary search for beta so the row's entropy matches the target."""
    lo, hi = -np.inf, np.inf
    beta = 1.0
    target = np.log(perplexity)
    for _ in range(max_iter):
        p = row_conditional(d2_i, beta, self_idx)
        s = p.sum()
        if s <= 0.0:
            beta /= 2.0
            continue
        h = np.log(s) + beta * (d2_i * p).sum() / s
        if abs(h - target) < tol:
            break
        if h > target:      # too spread out, needs a larger beta
            lo = beta
            beta = beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
        else:               # too peaked, needs a smaller beta
            hi = beta
            beta = beta / 2.0 if lo == -np.inf else (beta + lo) / 2.0
    p = row_conditional(d2_i, beta, self_idx)
    return p / p.sum(), beta


def joint_probabilities(D2, perplexity):
    """Symmetric P with zero diagonal, normalized to sum to one."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    betas = np.zeros(n)
    for i in range(n):
        d = D2[i].copy()
        d[i] = 1e12  # the self pair gets no probability mass
        row, beta = target_perplexity_row(d, perplexity, i)
        P[i] = row
        betas[i] = beta
    P = (P + P.T) / 2.0
    P /= P.sum()
    return P, betas
