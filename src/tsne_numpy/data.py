"""The sklearn digits set, standardized and PCA-reduced."""

import numpy as np
from sklearn.datasets import load_digits


def load():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y
