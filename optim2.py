import numpy as np
import time
from copy import deepcopy

import optim
from rpr import func_val, grad, stoc_grad, batch_grad, grad_lipshitz, linearized_grad, SimpleDataset


# SGD
def sgd(w, X, y, Xt, yt, rng, param_dict, num_epochs, avg=True, verbose=True):
    w_avg = deepcopy(w)
    n = y.shape[0]
    fac = 1.0 / n

    epochs = []
    fs = []
    fts = []

    def check_test(epoch=0):
        nonlocal epochs, fs, fts
        w_ = w_avg if avg else w
        f = func_val(X, y, w_)
        ft = func_val(Xt, yt, w_)
        epochs += [epoch]
        fs += [f]
        fts += [ft]
        if verbose:
            print('{:f}\t{:3.3f}'.format(f, ft))

    check_test()
    for epoch in range(num_epochs):
        t1 = time.time()
        for i in range(n):
            it = (epoch * n + i) // param_dict['t0'] + 1
            lr = param_dict['initial_lr'] * it ** (- param_dict['lr_decay'])
            idx = rng.randint(0, n)
            w -= lr * stoc_grad(X, y, w, idx)
            w_avg = w_avg * (1 - fac) + w * fac
        check_test(epoch + 1)
        if np.isinf(fs[-1]) or np.isnan(fs[-1]) or fs[-1] > 1e10:
            raise FloatingPointError('Nan or inf or too large')
    w_ret = w_avg if avg else w
    return w_ret, epochs, fs, fts


def prox_linear(w, X, y, Xt, yt, rng, param_dict, solver='svrg', num_epochs=500, verbose=True, option=True):
    if solver not in ['svrg', 'csvrg']:
        raise Exception('Unknown solver {}'.format(solver))
    train_dataset = SimpleDataset(X, y)
    test_dataset = SimpleDataset(Xt, yt)
    lr = param_dict['lr']
    mu = param_dict['mu']
    epochs = []
    fs = []
    fts = []

    solver_fn = optim.svrg if solver == 'svrg' else optim.catalyst_svrg

    def check_test(epoch=0):
        nonlocal epochs, fs, fts
        w_ = w
        f = func_val(X, y, w_)
        ft = func_val(Xt, yt, w_)
        epochs += [epoch]
        fs += [f]
        fts += [ft]
        if verbose:
            print('{:f}\t{:3.3f}'.format(f, ft))

    check_test()
    num_outer_epochs = num_epochs // param_dict['num_inner_epochs']
    for epoch in range(num_outer_epochs):
        Delta0 = np.zeros_like(w)
        grad_fn = lambda Delta, xi, yi: linearized_grad(xi, yi, w, Delta, mu)
        L = grad_lipshitz(X, y, w, mu)
        Delta = solver_fn(Delta0, None, grad_fn, train_dataset, test_dataset,
                          num_passes=param_dict['num_inner_epochs'],
                          L=L, l2reg=1.0 / lr,
                          seed=rng.randint(0, 100000), option=option)
        w += Delta
        check_test((epoch + 1) * param_dict['num_inner_epochs'])
        if np.isinf(fs[-1]) or np.isnan(fs[-1]) or fs[-1] > 1e10:
            raise FloatingPointError('Nan or inf or too large')
    return w, epochs, fs, fts


