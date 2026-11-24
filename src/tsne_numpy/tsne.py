"""The t-SNE objective and the optimizer that minimizes it."""

import numpy as np

from .affinity import joint_probabilities, sq_dists


def q_distribution(Y):
    """Student-t affinities with one degree of freedom, normalized."""
    d2 = sq_dists(Y)
    inv = 1.0 / (1.0 + d2)
    np.fill_diagonal(inv, 0.0)
    Q = inv / inv.sum()
    np.fill_diagonal(Q, 0.0)
    return Q, inv


def kl_divergence(P, Q):
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


def kl_gradient(Y, P, Q, inv):
    """dC/dY for the symmetric KL. inv holds 1/(1 + d2) already.

    Row i of the sum is 4 * sum_j A[i,j] (y_i - y_j) with
    A = (P - Q) * inv, which is 4 * (rowsum_i * y_i - A_i @ Y).
    """
    A = (P - Q) * inv
    rowsum = A.sum(axis=1)
    return 4.0 * (rowsum[:, None] * Y - A @ Y)


def fit(X, n_components=2, perplexity=30.0, pca_dim=50, n_iter=500,
        exaggeration=4.0, exaggeration_iters=100, lr=100.0, seed=2,
        verbose=False):
    """Returns (Y, history). PCA to pca_dim first when the input is wider."""
    X = np.asarray(X, dtype=np.float64)
    if pca_dim and X.shape[1] > pca_dim:
        mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        X = (X - mu) @ Vt[:pca_dim].T
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    P, _ = joint_probabilities(sq_dists(X), perplexity)
    Y = rng.normal(scale=1e-4, size=(n, n_components))
    Y_old = Y.copy()
    history = []
    for it in range(1, n_iter + 1):
        Q, inv = q_distribution(Y)
        cost = kl_divergence(P, Q)
        history.append(cost)
        grad = kl_gradient(Y, P, Q, inv)
        if it <= exaggeration_iters:
            grad *= exaggeration
        # momentum picks up after the early phase, as in the paper
        momentum = 0.5 if it < 250 else 0.8
        step = -grad * lr
        Y_new = Y + momentum * (Y - Y_old) + step
        Y_old, Y = Y, Y_new
        if verbose and (it % 100 == 0 or it == 1):
            print(f"iter {it:4d}  KL {cost:.4f}")
    return Y, history
