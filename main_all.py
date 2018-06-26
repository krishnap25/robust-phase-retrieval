import numpy as np
from numpy import linalg as LA
import time, sys, os
import argparse
import pickle as pkl
from copy import deepcopy

from data import generate_dataset
from optim2 import sgd, prox_linear

# Argument parser
parser = argparse.ArgumentParser(description='Run robust phase retrieval.')
parser.add_argument('--cond', type=float, default=1,  help='condition number')
parser.add_argument('--noise', type=float, default=1, help='noise scale')

if __name__ == '__main__':
    args = parser.parse_args()
    np.seterr(all='raise')
    # Generate data
    n, nt, d = 500, 200, 50
    X, y, Xt, yt = generate_dataset(args.cond, args.noise, n=n, nt=nt, d=d)
    # Params
    SEEDS = list(range(1))
    INIT_SCALES = [1e-3, 1e-2, 1e-1]
    INIT_LRS_SGD = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # INIT_LRS_SGD = [1e-4]
    LR_DECAYS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    # LR_DECAYS = [0.5]
    MUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # MUS = [1e-4]
    NUM_INNER_EPOCHS = [1, 2, 3, 5]
    PL_LRs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # PL_LRs = [1e-3]
    CSVRG_OPTIONS = [1, 2, 3]

    num_epochs = 500
    verbose = False

    t1 = time.time()
    # SGD
    print('Starting SGD')
    for init_scale in INIT_SCALES:
        for init_lr in INIT_LRS_SGD:
            for lr_decay in LR_DECAYS:
                for avg in [True, False]:
                    try:
                        for seed in SEEDS:
                            fn = 'outs/sgd_{}_{}_{}_lr_{}_{}_{}_{}.p'.format(
                                args.cond, args.noise, init_scale, init_lr, lr_decay, avg, seed
                                                 )
                            rng = np.random.RandomState(seed)
                            w0 = rng.randn(d) * init_scale
                            param_dict = {'initial_lr': init_lr, 'lr_decay': lr_decay, 't0': 1,
                                          'init_scale': init_scale, 'avg': avg,
                                          'seed': seed,
                                          'cond': args.cond, 'noise': args.noise}
                            w, epochs, fs, fts = sgd(w0, X, y, Xt, yt, rng, param_dict,
                                                     num_epochs=num_epochs, avg=avg,
                                                     verbose=verbose)
                            with open(fn, 'wb') as fp:
                                pkl.dump([w, epochs, fs, fts, param_dict], fp)
                    except FloatingPointError as e:
                        print(e, param_dict)
                        continue
    print('SGD time: ', time.time() - t1)

    t1 = time.time()
    # SVRG
    print('Starting SVRG')
    for init_scale in INIT_SCALES:
        for lr in PL_LRs:
            for mu in MUS:
                for num_inner_epochs in NUM_INNER_EPOCHS:
                    try:
                        for seed in SEEDS:
                            fn = 'outs/pl_svrg_{}_{}_{}_lr_{}_{}_{}_{}.p'.format(
                                args.cond, args.noise, init_scale, lr, mu, num_inner_epochs, seed
                                                 )
                            rng = np.random.RandomState(seed)
                            w0 = rng.randn(d) * init_scale
                            param_dict = {'lr': lr, 'mu': mu,
                                          'init_scale': init_scale,
                                          'num_inner_epochs': num_inner_epochs,
                                          'seed': seed,
                                          'cond': args.cond, 'noise': args.noise}
                            w, epochs, fs, fts = prox_linear(w0, X, y, Xt, yt, rng, param_dict,
                                                             solver='svrg', num_epochs=num_epochs,
                                                             verbose=verbose)
                            with open(fn, 'wb') as fp:
                                pkl.dump([w, epochs, fs, fts, param_dict], fp)
                    except FloatingPointError:
                        continue
    print('SVRG time: ', time.time() - t1)


    t1 = time.time()
    # Catalyst SVRG
    print('Starting C-SVRG')
    for init_scale in INIT_SCALES:
        for lr in PL_LRs:
            for mu in MUS:
                for num_inner_epochs in NUM_INNER_EPOCHS:
                    for option in CSVRG_OPTIONS:
                        try:
                            for seed in SEEDS:
                                fn = 'outs/pl_csvrg_{}_{}_{}_lr_{}_{}_{}_option{}_{}.p'.format(
                                    args.cond, args.noise, init_scale, lr, mu,
                                    num_inner_epochs, option, seed
                                )
                                rng = np.random.RandomState(seed)
                                w0 = rng.randn(d) * init_scale
                            param_dict = {'lr': lr, 'mu': mu,
                                          'init_scale': init_scale,
                                          'num_inner_epochs': num_inner_epochs,
                                          'seed': seed, 'option': option,
                                          'cond': args.cond, 'noise': args.noise}
                                w, epochs, fs, fts = prox_linear(w0, X, y, Xt, yt, rng, param_dict,
                                                                 solver='csvrg', num_epochs=num_epochs,
                                                                 verbose=verbose)
                                with open(fn, 'wb') as fp:
                                    pkl.dump([w, epochs, fs, fts, param_dict], fp)
                        except FloatingPointError:
                            continue
    print('SVRG time: ', time.time() - t1)

