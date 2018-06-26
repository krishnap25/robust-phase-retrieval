import numpy as np
import time
from copy import deepcopy
import sys


def batch_average(fun, w, dataset, it):
    # fun: function that returns a scalar or a np.array
    x, y = dataset[0]
    ret = fun(w, x, y)
    for i in range(1, len(dataset)):
        x, y = dataset[i]
        ret += fun(w, x, y) # TODO
    ret /= len(dataset)
    return ret


# SGD
def sgd(w0, loss_fn, grad_fn, train_dataset, test_dataset, num_passes=10, lr=1.0,
        l2reg=0.0, seed=1, eval_fn=None):
    # w0: np.array of shape [dim]
    # loss_fn, grad_fn: callable. args: (w, x, y, iteration) where (x, y) <- dataset[i]
    # dataset: reader.Dataset object
    rng = np.random.RandomState(seed)
    w = w0
    w_avg = deepcopy(w0)
    count = 0.0

    t = time.time()
    for epoch in range(num_passes):
        t1 = time.time()
        for i in range(len(train_dataset)):
            iteration = epoch * len(train_dataset) + i
            idx = rng.randint(0, len(train_dataset))
            x, y = train_dataset[idx]
            grad = grad_fn(w, x, y)
            w *= (1 - lr * l2reg)
            w -= lr * grad
            # average
            w_avg *= count / (count + 1)
            w_avg += w / (count + 1)
            count += 1
        # metrics
    return w_avg 



# SVRG

class _SVRG:
    def __init__(self, loss_fn, grad_fn, w0, lr, l2reg, n):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.l2reg = l2reg
        self.lr = lr
        self.n = n
        # weight vectors
        self.w0 = w0
        self.w = deepcopy(w0)
        self.w_avg = deepcopy(w0)
        self.idx = 0.0
        self.stable_grad = None

    def update_stable_grad(self, g):
        self.stable_grad = g

    def update_iterates(self):
        self.idx = 0.0
        np.copyto(self.w0, self.w_avg)
        np.copyto(self.w, self.w0)

    def step(self, X, y, iteration):
        assert self.stable_grad is not None
        grad = self.grad_fn(self.w, X, y)
        grad0 = self.grad_fn(self.w0, X, y)
        # update = self.l2reg * self.w + self.kappa * (self.w - self.y)  + grad - grad0 + self.stable_grad
        # self.w -= (self.lr * update)
        self.w *= (1 - self.lr * self.l2reg)
        self.w -= self.lr * (grad - grad0 + self.stable_grad)
        # update avg
        i = self.idx
        self.w_avg *= i / (i + 1.0)
        self.w_avg += self.w / (i + 1)
        self.idx += 1


def svrg(w0, loss_fn, grad_fn, train_dataset, test_dataset,
         num_passes=10, lr=None, L=None, l2reg=0, seed=1, eval_fn=None, option=None):
    if lr is None and L is None:
        raise Exception('Require one of lr or L to not be None')
    lr = lr if lr is not None else 1.0 / (L + l2reg)
    rng = np.random.RandomState(seed)
    n = len(train_dataset)
    optimizer = _SVRG(loss_fn, grad_fn, w0, lr, l2reg, n)
    count = 0.0
    grad = batch_average(grad_fn, optimizer.w_avg, train_dataset, 0)
    optimizer.update_stable_grad(grad)

    if num_passes < 1:
        nn = int(num_passes * n)
        num_passes = 1
    else:
        nn = n
        num_passes = int(num_passes)
    # grad norm doesn't work because scipy.linalg.norm. TODO.
    for epoch in range(num_passes):
        t1 = time.time()
        for i in range(nn):
            iteration = epoch * len(train_dataset) + i
            idx = rng.randint(0, n)
            x, y = train_dataset[idx]
            optimizer.step(x, y, iteration)

        optimizer.update_iterates()
        grad = batch_average(grad_fn, optimizer.w0, train_dataset, 0)
        optimizer.update_stable_grad(grad)
    return optimizer.w_avg


# Catalyst SVRG helper

class _CatalystSVRG:
    def __init__(self, loss_fn, grad_fn, w0, L, l2reg, n, kappa=1, option=2):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.l2reg = l2reg
        self.L = L
        self.n = n
        self.option = option
        # kappa, q, alpha
        if self.L / (self.n + 1) > self.l2reg:
            self.kappa = (self.L / (self.n + 1) - self.l2reg) * kappa
        else:
            self.kappa = self.L / (self.n + 1) * kappa
        self.lr = 1 / (self.L + self.l2reg + self.kappa)

        self.q = self.l2reg / (self.l2reg + self.kappa)
        if self.l2reg > 0:
            self.alpha = np.sqrt(self.q)
        else:
            self.alpha = 1.0
        # weight vectors
        self.w0 = w0
        self.w = deepcopy(w0)
        self.w_avg = deepcopy(w0)
        self.w_avg_prev = deepcopy(w0)
        self.y = deepcopy(w0)
        self.y_prev = deepcopy(w0)
        self.idx = 0.0
        self.stable_grad = None

    def update_stable_grad(self, g):
        self.stable_grad = g

    def update_iterates(self):
        self.idx = 0.0
        # alpha and beta
        q, alpha = self.q, self.alpha
        alpha_new = 0.5 * (q - alpha**2) + 0.5 * \
            np.sqrt(4 * alpha**2 + (q - alpha**2)**2)
        beta = alpha * (1 - alpha) / (alpha**2 + alpha_new)
        self.alpha = alpha_new
        # self.y_prev = self.y
        np.copyto(self.y_prev, self.y)
        # self.y = self.w_avg + beta * (self.w_avg - self.w_avg_prev)
        np.copyto(self.y, self.w_avg)
        self.y *= (1 + beta)
        self.y -= beta * self.w_avg_prev
        # self.w_avg_prev = self.w_avg
        np.copyto(self.w_avg_prev, self.w_avg)
        # option 1: w0 = y
        # option 2: w0 = w_avg + self.kappa / (self.kappa + self.l2reg) * (y - y_prev)
        # option 2:
        if self.option == 2:
            temp = self.kappa / (self.kappa + self.l2reg)
            np.copyto(self.w0, self.w_avg)
            self.w0 += temp * (self.y - self.y_prev)
        elif self.option == 1:
            np.copyto(self.w0, self.y)
        else:
            np.copyto(self.w0, self.w_avg)
        np.copyto(self.w, self.w0)

    def step(self, X, y, iteration):
        assert self.stable_grad is not None
        grad = self.grad_fn(self.w, X, y)
        grad0 = self.grad_fn(self.w0, X, y)
        # update = self.l2reg * self.w + self.kappa * (self.w - self.y)  + grad - grad0 + self.stable_grad
        # self.w -= (self.lr * update)
        self.w *= (1 - self.lr * self.l2reg - self.lr * self.kappa)
        self.w += self.lr * self.kappa * self.y
        self.w -= self.lr * (grad - grad0 + self.stable_grad)
        # update avg
        i = self.idx
        self.w_avg *= i / (i + 1.0)
        self.w_avg += self.w / (i + 1)
        self.idx += 1

# run Catalyst-SVRG


def catalyst_svrg(w0, loss_fn, grad_fn, train_dataset, test_dataset,
                  num_passes=10, L=1, l2reg=0, seed=1, eval_fn=None, option=2):
    rng = np.random.RandomState(seed)
    n = len(train_dataset)
    optimizer = _CatalystSVRG(loss_fn, grad_fn, w0, L, l2reg, n, option=option)
    count = 0.0
    iteration = 0
    grad = batch_average(grad_fn, optimizer.w_avg, train_dataset, iteration)

    optimizer.update_stable_grad(grad)
    for epoch in range(num_passes):
        t1 = time.time()
        for i in range(n):
            iteration = epoch * len(train_dataset) + i
            idx = rng.randint(0, n)
            x, y = train_dataset[idx]
            optimizer.step(x, y, iteration)

        optimizer.update_iterates()
        grad = batch_average(grad_fn, optimizer.w0, train_dataset, iteration)
        optimizer.update_stable_grad(grad)
    return optimizer.w_avg


