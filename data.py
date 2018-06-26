import numpy as np
from numpy import linalg as LA
from scipy.stats import ortho_group


def generate_dataset(cond_num, noise_scale, n=500, nt=200, d=50):
    rng = np.random.RandomState(1)
    w_star = rng.randn(d)
    w_star /= np.linalg.norm(w_star)

    # hyper param
    R = np.diag(np.linspace(1, cond_num, num=d))

    # Train

    # sample random orthogonal matrix
    m = ortho_group.rvs(dim=n, random_state=rng)
    U = m[:, :d]
    # print(np.trace(U.T.dot(U)))
    # print(LA.eig(U.T.dot(U))[0])

    X = np.matmul(U, R) # n x d data matrix
    noise = rng.laplace(scale=noise_scale, size=(n,))
    y = np.dot(X, w_star) ** 2 + noise

    # Test
    m = ortho_group.rvs(dim=nt, random_state=rng)
    U = m[:, :d]
    # print(np.trace(U.T.dot(U)))
    # print(LA.eig(U.T.dot(U))[0])

    Xt = np.matmul(U, R) # n x d data matrix
    noise = rng.laplace(scale=noise_scale, size=(nt,))
    yt = np.dot(Xt, w_star) ** 2 + noise
    return X, y, Xt, yt
