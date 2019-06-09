#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:28:14 2017

@author: alain
"""

from time import time as time
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
from screening_lasso import generate_random_gaussian
from noncvx_lasso import MMLasso, MMLasso_screening, \
    MMLasso_screening_genuine, BCD_noncvxlasso_lsp, check_opt_logsum, \
    approx_lsp
from noncvx_lasso import current_cost, reg_lsp, prox_lsp
from noncvx_lasso import GIST

import warnings
warnings.filterwarnings("ignore")

max_iter = 2000
maxiter_inner = 100
init_iter = 0



def run_BCD(X, y, lambd, theta, tol=1e-5, w_init=[]):
    tic = time()
    w_bcd = BCD_noncvxlasso_lsp(X, y, lambd, theta, tol=tol, max_iter=max_iter,
                                w_init=w_init)
    time_bcd = time() - tic
    return w_bcd, time_bcd


def run_GIST(X, y, lambd, theta, tol=1e-5, w_init=[]):
    """
    generalized ISTA
    """
    tic = time()
    w_gist, _ = GIST(X, y, lambd, theta, reg=reg_lsp, prox=prox_lsp, eta=1.5,
                     sigma=0.1, tol=tol, max_iter=max_iter, w_init=w_init)
    time_gist = time() - tic
    return w_gist, time_gist


def run_MMscreening_genuine(X, y, lambd, theta, tol=1e-5, dual_gap_inner=1e-3,
                            w_init=[]):
    """
    majorization minimization only with inner screening
    """
    tic = time()
    w_mmlsp_screened_gen = MMLasso_screening_genuine(X, y, lambd, theta,
                                                     approx=approx_lsp,
                                                     maxiter=max_iter,
                                                     tol_first_order=tol,
                                                     dual_gap_inner=dual_gap_inner,
                                                     maxiter_inner=maxiter_inner,
                                                     w_init=w_init)
    time_mmlsp_screened_gen = time() - tic
    return w_mmlsp_screened_gen, time_mmlsp_screened_gen


def run_MMscreening_2(X, y, lambd, theta, tol=1e-5, dual_gap_inner=1e-3,
                      screen_frq=2, w_init=[]):
    """
    Majorization-minimization with inner screening and screening propagation
    """
    tic = time()
    w_mmlsp_screened = MMLasso_screening(X, y, lambd, theta, approx=approx_lsp,
                                         maxiter=max_iter, initial_screen=True,
                                         method=2, screen_frq=screen_frq,
                                         tol_first_order=tol,
                                         dual_gap_inner=dual_gap_inner,
                                         maxiter_inner=maxiter_inner,
                                         w_init=w_init, algo_method='bcd',
                                         init_iter=init_iter)
    time_mmlsp_screened = time() - tic
    return w_mmlsp_screened, time_mmlsp_screened


def run_algo(X, y, lambd, theta, algo, tol, dual_gap_inner=1e-3,
             screen_every=2, w_init=[]):

    if algo == 'bcd':
        w, run_time = run_BCD(X, y, lambd, theta, tol=tol, w_init=w_init)
    elif algo == 'gist':
        w, run_time = run_GIST(X, y, lambd, theta, tol=tol, w_init=w_init)

    elif algo == 'mm_screening_genuine':
        w, run_time = run_MMscreening_genuine(X, y, lambd, theta,
                                              tol=tol,
                                              dual_gap_inner=dual_gap_inner,
                                              w_init=w_init)

    elif algo == 'mm_screening_2':
        w, run_time = run_MMscreening_2(X, y, lambd, theta,
                                        screen_frq=screen_frq, tol=tol,
                                        dual_gap_inner=dual_gap_inner,
                                        w_init=w_init)

    return w, run_time


def compute_performance(w, lambd, theta, wopt, reg=reg_lsp, tol=1e-3):
    optimality = check_opt_logsum(X, y, w, lambd, theta, tol=tol)
    maxi = np.max(abs(wopt - w))
    y_true = np.abs(wopt) > 0
    y_pred = np.abs(w) > 0
    f_meas_a = precision_recall_fscore_support(y_true, y_pred, pos_label=1,
                                               average='binary')[2]
    cost = current_cost(X, y, w, lambd, theta, reg)
    return optimality, maxi, f_meas_a, cost

if __name__ == '__main__':
    
    n_iter = 1
    n_samplesvec =  [50]   # you can replace this value with 500 and
    n_featuresvec = [100]  # 5000 but i would take several days unless 
                           # you use one thread for one algo     
    n_informativevec = [5]

    sigma_bruit = 2        # settting 0.01, you can reproduce gain for bcd,
                           # MM_genuine and MM_screening 


    N = 50
    Tvec = np.power(10.0, -3 * np.linspace(1, N - 1, N) / (N - 1))
    thetavec = [1, 0.1, 0.01]


    tolvec = [1e-3,1e-4,1e-5]
    dual_gap_inner = 1e-4
    screen_frq = 10
    path_results = './'
    algo_list = ['bcd', 'gist', 'mm_screening_genuine', 'mm_screening_2']

    
    dual_gap_inner = 1e-4
    
    for tol in tolvec :
        for algo in algo_list:   
            print('running {}'.format(algo))
            for n_samples, n_features in zip(n_samplesvec, n_featuresvec):
                    for n_informative in n_informativevec:
                        filename = path_results + '{}_n_samples{:d}_n_feat{:d}_n_inform{:d}_bruit{:2.2f}_N{:d}_tol{:2.5e}'.format(algo,n_samples,n_features,n_informative,sigma_bruit,N,tol)
                        opt_mm = 'gap_{:1.0e}_screen_{:d}'.format(dual_gap_inner,
                                                                  screen_frq)
                        filename = filename + opt_mm
                        print(filename)
                        timing = np.zeros([len(Tvec), len(thetavec), n_iter])
                        optimality = np.zeros([len(Tvec), len(thetavec), n_iter])
                        maxi = np.zeros([len(Tvec), len(thetavec), n_iter])
                        f_meas = np.zeros([len(Tvec), len(thetavec), n_iter])
                        cost = np.zeros([len(Tvec), len(thetavec), n_iter])
        
                        for i in range(n_iter):
                            np.random.seed(i)
                            X, y, wopt = generate_random_gaussian(n_samples,
                                                                  n_features,
                                                                  n_informative,
                                                                  sigma_bruit)
                            lambdamax = np.max(np.abs(np.dot(X.transpose(), y)))
                            lambdavec = lambdamax * Tvec
                            print(lambdavec)
                            for i_theta, theta in enumerate(thetavec):
                                lambdavec /= np.log(1 + 1 / theta)
                                if i_theta == 0:
                                    w_init = []
                                else:
                                    w_init = w_th.copy()
                                for i_lambd, lambd in enumerate(lambdavec):
                                    w, run_time = run_algo(X, y, lambd, theta, algo,
                                                           tol, dual_gap_inner,
                                                           screen_frq, w_init)
                                    w_init = w.copy()
                                    if i_lambd == 0:
                                        w_th = w.copy()
                                    timing[i_lambd, i_theta, i] = run_time
                                    tol_opt = max(tol, 1e-3)
                                    optimality[i_lambd, i_theta, i], maxi[i_lambd, i_theta, i], f_meas[i_lambd, i_theta, i], cost[i_lambd, i_theta, i] = compute_performance(w,lambd, theta,wopt,tol=tol_opt)
                                    print(i, theta, lambd, run_time,
                                          np.sum(timing[:, :, i]),
                                          f_meas[i_lambd, i_theta, i],
                                          optimality[i_lambd, i_theta, i])
                            np.savez(filename, timing=timing, optimality=optimality,
                                     maxi=maxi, f_meas=f_meas, cost=cost)
