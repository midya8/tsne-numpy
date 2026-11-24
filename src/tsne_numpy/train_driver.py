"""Runs t-SNE on the digits set, scores it against PCA, saves results."""

import json
import os

import numpy as np

from .affinity import sq_dists
from .data import load
from .eval import knn_overlap, trustworthiness
from .tsne import fit

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    X, y = load()
    Y, history = fit(X, n_iter=500, perplexity=30.0, seed=2)
    Do = sq_dists(X)
    De = sq_dists(Y)
    results = {"n": int(X.shape[0]), "perplexity": 30.0, "iters": 500,
               "kl_final": round(history[-1], 4),
               "kl_iter100": round(history[99], 4),
               "knn_overlap_tsne": round(knn_overlap(Do, De), 4),
               "knn_overlap_pca2": None,
               "trustworthiness_tsne": round(trustworthiness(X, Y), 4)}
    # the PCA-2 baseline, scored the same way
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Yp = (X - mu) @ Vt[:2].T
    Dep = sq_dists(Yp)
    results["knn_overlap_pca2"] = round(knn_overlap(Do, Dep), 4)
    results["trustworthiness_pca2"] = round(trustworthiness(X, Yp), 4)
    with open(os.path.join(ASSETS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].scatter(Yp[:, 0], Yp[:, 1], c=y, s=8, cmap="tab10")
    axes[0].set_title("PCA, 2 components")
    axes[1].scatter(Y[:, 0], Y[:, 1], c=y, s=8, cmap="tab10")
    axes[1].set_title("t-SNE, this repo")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_tsne.png"), dpi=110)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
