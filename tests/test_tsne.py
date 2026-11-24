import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsne_numpy.affinity import joint_probabilities, sq_dists, target_perplexity_row
from tsne_numpy.eval import knn_overlap, trustworthiness
from tsne_numpy.tsne import fit, kl_divergence, kl_gradient, q_distribution


def small_data(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(15, 5)) + np.array([6, 0, 0, 0, 0])
    b = rng.normal(size=(15, 5)) + np.array([-6, 0, 0, 0, 0])
    return np.vstack([a, b])


def test_joint_probabilities_are_a_distribution():
    X = small_data()
    P, betas = joint_probabilities(sq_dists(X), 10.0)
    assert abs(P.sum() - 1.0) < 1e-9
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 0.0)
    assert (P >= 0).all()


def test_perplexity_matches_the_target():
    X = small_data()
    D2 = sq_dists(X)
    for i in range(X.shape[0]):
        d = D2[i].copy()
        d[i] = 1e12
        row, _ = target_perplexity_row(d, 10.0, i)
        h = -(row * np.log(np.maximum(row, 1e-12))).sum()
        assert abs(np.exp(h) - 10.0) < 0.05


def test_q_distribution_is_a_distribution():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(20, 2))
    Q, inv = q_distribution(Y)
    assert abs(Q.sum() - 1.0) < 1e-9
    assert np.allclose(np.diag(Q), 0.0)


def test_kl_gradient_by_finite_differences():
    X = small_data()
    P, _ = joint_probabilities(sq_dists(X), 8.0)
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(30, 2)) * 0.5
    Q, inv = q_distribution(Y)
    grad = kl_gradient(Y, P, Q, inv)

    def loss(YY):
        Qq, _ = q_distribution(YY)
        return kl_divergence(P, Qq)

    eps = 1e-6
    checked = range(0, 30, 3)
    num = np.zeros((len(checked), 2))
    for r, i in enumerate(checked):
        for j in range(2):
            up, dn = Y.copy(), Y.copy()
            up[i, j] += eps
            dn[i, j] -= eps
            num[r, j] = (loss(up) - loss(dn)) / (2 * eps)
    # only the perturbed coordinates carry a numerical gradient
    assert np.max(np.abs(num - grad[list(checked)])) < 1e-5 * (
        1.0 + np.max(np.abs(grad)))


def test_fit_clusters_separable_groups():
    X = small_data()
    Y, history = fit(X, pca_dim=None, n_iter=120, perplexity=8.0, lr=50.0)
    assert history[-1] < history[0]
    # the two groups should sit apart from each other, in any orientation
    c0, c1 = Y[:15].mean(axis=0), Y[15:].mean(axis=0)
    between = np.linalg.norm(c0 - c1)
    spread = 0.5 * (np.std(Y[:15], axis=0).mean() + np.std(Y[15:], axis=0).mean())
    assert between > 2.0 * spread


def test_trustworthiness_matches_sklearn():
    from sklearn.manifold import trustworthiness as sk_trust
    X = small_data()
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(30, 2))
    mine = trustworthiness(X, Y, k=5)
    theirs = float(sk_trust(X, Y, n_neighbors=5))
    assert abs(mine - theirs) < 1e-6


def test_knn_overlap_bounds():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 6))
    D = sq_dists(X)
    assert knn_overlap(D, D) == 1.0
    shuffled = X[rng.permutation(40)]
    assert knn_overlap(D, sq_dists(shuffled)) < 1.0
