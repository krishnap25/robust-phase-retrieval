import numpy as np
from numpy import linalg as LA


def func_val(X, y, w):
    n = X.shape[0]
    return np.abs(np.dot(X, w) ** 2 - y).sum() / (2 * n)


def grad(xi, yi, w):
    return np.dot(xi, w) * xi * np.sign(np.dot(xi, w)**2 - yi)


def stoc_grad(X, y, w, idx):
    return grad(X[idx, :], y[idx], w)


def batch_grad(X, y, w):
    n, d = X.shape
    g = np.zeros(d)
    for i in range(n):
        g += stoc_grad(X, y, w, i) / n
    return g


def grad_lipshitz(X, y, w, mu):  # does not account for regularization
    return np.abs(np.matmul(X, w) * LA.norm(X, axis=1)).max() / mu


def linearized_grad(xi, yi, w, Delta, mu):
    dot = np.dot(xi, w)
    inner = (dot ** 2 + 2 * dot * np.dot(xi, Delta) - yi)
    temp = np.sign(inner) if abs(inner) > mu/2 else 4 * inner / mu
    g = dot * xi * temp
    return g  # Does not account for regularization


class SimpleDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


# For Debugging
def batch_grad_fd(X, y, w, eps=1e-5):
    d = w.shape[0]
    w1 = deepcopy(w)
    w2 = deepcopy(w)
    g = np.zeros(d)
    for i in range(d):
        w1[i] += eps
        w2[i] -= eps
        g[i] = (func_val(X, y, w1) - func_val(X, y, w2)) / (2 * eps)
        w1[i] -= eps
        w2[i] += eps
    return g
