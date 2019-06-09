# -*- coding: utf-8 -*-

from time import process_time as time
from numpy.linalg import norm

import numpy as np
from screening_lasso import generate_random_gaussian, \
     weighted_prox_lasso_bcd_screening
from screening_lasso import compute_duality_gap_lasso, \
    compute_duality_gap_prox_lasso


# Functions for LOGSUM regularizers

def reg_lsp(w, theta):
    """Compute the regularizer objective value."""
    return sum(np.log(1 + np.abs(w) / theta))


def subdiff_concavelsp(w, lbd, theta):
    """Compute the subgradient of the concave part.

    pen is : |w| - log(|w| + theta) including the regularizer strenght lbd.
    """
    return lbd * (1 - 1 / (abs(w) + theta))


def prox_lsp(w, lbd, theta):
    """Compute proximal operator of non-cvx regularizer (closed-form)."""
    absw = np.abs(w)
    z = absw - theta
    v = z * z - 4 * (lbd - absw * theta)
    v = np.maximum(v, 0)

    sqrtv = np.sqrt(v)
    w1 = np.maximum((z + sqrtv) / 2, 0)
    w2 = np.maximum((z - sqrtv) / 2, 0)
    # Evaluate the proximal at this solution
    y0 = 0.5 * w**2
    y1 = 0.5 * (w1 - absw)**2 + lbd * np.log(1 + w1 / theta)
    y2 = 0.5 * (w2 - absw)**2 + lbd * np.log(1 + w2 / theta)

    sel1 = (y1 < y2) & (y1 < y0)
    sel2 = (y2 < y1) & (y2 < y0)
    wopt = w1 * sel1 + w2 * sel2

    return np.sign(w) * wopt


def approx_lsp(w, theta):
    """Vector of the linear approximation of log(1+w/theta)."""
    return 1. / (np.abs(w) + theta)


def check_opt_logsum(X, y, w, lbd, theta, tol=1e-3, tol_val=1e-4):
    """Evaluate first-order optimality conditions at tol non-zeros values."""
    if type(lbd) is not np.array:
        lbd = np.full(X.shape[1], lbd)
    if type(theta) is not np.array:
        theta = np.full(X.shape[1], theta)
    residual = y - X.dot(w)
    correl = X.T.dot(residual)
    absw = np.abs(w)
    ind_nz = np.where(absw > tol_val)[0]
    ind_zero = np.where(absw < tol_val)[0]
    if ind_zero.shape[0] > 0:
        opt_ind_zero = np.all(abs(- correl[ind_zero]) <=
                              (lbd[ind_zero] / theta[ind_zero] + tol))
    else:
        opt_ind_zero = True
    if ind_nz.shape[0] > 0:
        opt_ind_nz = np.all(abs(- correl[ind_nz] + lbd[ind_nz] * np.sign(w[ind_nz]) / (theta[ind_nz] + np.abs(w[ind_nz]))) < tol)
    else:
        opt_ind_nz = True

    return opt_ind_zero and opt_ind_nz

#
# Generic functions
#

def current_cost(X, y, w, lbd, theta, reg=reg_lsp):
    """Compute the objective function 
    
        min_w 0.5 || y - X@w||_2^2 + \lambda*reg
        
        where reg is nonconvex regularizer
    """
    normres2 = norm(y - X.dot(w))**2
    return 0.5 * normres2 + lbd * reg(w, theta)


def GIST(X, y, lbd, theta, reg=reg_lsp, prox=prox_lsp, eta=1.5,
         sigma=0.1, tol=1e-3, max_iter=1000, w_init=[], tmax=1e20):
    """
    Solve 
    
    min_w 0.5 || y - X@w||_2^2 + \lambda*reg
        
    where reg is the non-convex regularizer using an 
    iterative shrinkage and thresholding strategy. 
    
    see Gong et al.  A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
    Regularized Optimization Problems
    
    
    
    """

    multi = 5
    n_features = X.shape[1]
    if w_init == []:
        wp = np.zeros(n_features)
    else:
        wp = w_init
    cout = np.zeros(max_iter * multi)
    coutnew = current_cost(X, y, wp, lbd, theta, reg=reg)
    for i in range(max_iter * multi):
        t = 1
        grad = - X.T.dot(y - X.dot(wp))
        wp_aux = prox(wp - grad / t, lbd / t, theta)
        # backtracking stepsize
        coutold = coutnew
        coutnew = current_cost(X, y, wp_aux, lbd, theta, reg=reg)
        while coutnew - coutold > - sigma / 2 * t * norm(wp - wp_aux)**2:
            t = t * eta
            wp_aux = prox(wp - grad / t, lbd / t, theta)
            coutnew = current_cost(X, y, wp_aux, lbd, theta, reg=reg)
            if t > tmax:
                print('not converging. too small step')
                break
        cout[i] = coutnew

        wp = wp_aux.copy()
        # testing optimality
        if prox == prox_lsp:
            opt = check_opt_logsum(X, y, wp, lbd, theta, tol=tol)
            if opt:
                break
    return wp, cout


def BCD_noncvxlasso_lsp(X, y, lbd, theta, max_iter=500, tol=1e-6,
                        w_init=[], screen_frq=2):
    """ 
    Solve 
    
    min_w 0.5 || y - X@w||_2^2 + \lambda*reg
        
    where reg is the non-convex regularizer using a non-convex coordinate
    descent strategy
    
    """
    
    
    n_features = X.shape[1]
    if w_init == []:
        w = np.zeros(n_features)
    else:
        w = w_init

    i = 0
    opt = False
    screened = np.zeros(n_features)
    rho = y - X.dot(w)

    while i < max_iter and not opt:
        for j in range(n_features):
            if not screened[j]:
                xj = X[:, j]
                s = rho + xj * w[j]
                xts = xj.T.dot(s)
                # logsum resolution
                # equation 2nd degré
                p = np.zeros(3)
                p[0] = xj.dot(xj)
                p[1] = xj.dot(xj) * theta - xts
                p[2] = lbd - theta * xts
                racine = np.roots(p)
                racine = np.append(racine, 0)
                cout_aux_pos = [(np.real(waa),
                                 np.real(0.5 * np.sum((s - xj * waa)**2) + lbd * np.log(1 + waa / theta))) for i, waa in enumerate(racine) if not np.iscomplex(waa) and np.real(waa)>=0]
                p[1] = -(xj.dot(xj)) * theta - xts
                p[2] = lbd + theta * xts
                racine = np.roots(p)
                cout_aux_neg = [(np.real(waa),
                                 np.real(0.5 * np.sum((s - xj * waa)**2) + lbd * np.log(1 - waa / theta))) for i, waa in enumerate(racine) if not np.iscomplex(waa) and np.real(waa)<0]
                cout_aux = cout_aux_pos + cout_aux_neg
                cout_aux.sort(key=lambda tup: tup[1])
                wp = cout_aux[0][0]
                rho = rho - xj * (wp - w[j])
                w[j] = wp

        opt = check_opt_logsum(X, y, w, lbd, theta, tol)

        i += 1
    return w


def MMLasso(X, y, lbd, theta, approx=approx_lsp, maxiter=1000,
            tol_first_order=1e-3, dual_gap_inner=1e-2,
            maxiter_inner=1000, w_init=[]):

    """
    naive majorizartion minimization strategy without any inner screening
    nor sceening propagation
    """

    n_features = X.shape[1]
    if w_init == []:
        w_mm = np.full(n_features, 0.)
    else:
        w_mm = w_init.copy()
    cout_mm = []
    if type(lbd) is not np.array:
        lbd = np.full(X.shape[1], lbd)
    opt = False
    i = 0
    while i < maxiter and not opt:
        lbdaux = lbd * approx(w_mm, theta)

        w_mm, _, _, _, _ = weighted_prox_lasso_bcd_screening(X, y, lbdaux, nbitermax=maxiter_inner,
                                  dual_gap_tol=dual_gap_inner, winit=w_mm, do_screen=False)
        if approx == approx_lsp:
            opt = check_opt_logsum(X, y, w_mm, lbd, theta, tol=tol_first_order)
        i += 1
    return w_mm, cout_mm


def MMLasso_screening_genuine(X, y, lbd, theta, approx=approx_lsp,
                              maxiter=1000, tol_first_order=1e-3,
                              dual_gap_inner=1e-2, screen_frq_inner=5,
                              maxiter_inner=1000, w_init=[], init_iter=5):
    """ Majorization-minimization using weightedLasso with inner screening"""
    n_features = X.shape[1]
    if w_init == []:
        w_mm = np.full(n_features, 0.)

    else:
        w_mm = w_init
    if type(lbd) is not np.array:
        lbd = np.ones(X.shape[1]) * lbd
    opt = False
    i = 0

    while i < maxiter and not opt:
        lbdaux = lbd * approx(w_mm, theta)

        if i < init_iter:
            dual_gap = dual_gap_inner
        else:
            dual_gap = dual_gap_inner

        w_mm, hist_screen, _, residual, correl = weighted_prox_lasso_bcd_screening(X, y, lbdaux, w_mm,
                                    winit=w_mm, dual_gap_tol=dual_gap,
                                    screen_frq=screen_frq_inner,
                                    nbitermax=maxiter_inner)
        if approx == approx_lsp:
            opt = check_opt_logsum(X, y, w_mm, lbd, theta, tol_first_order)

        i += 1
    return w_mm


def MMLasso_screening(X, y, lbd, theta, approx=approx_lsp, maxiter=1000,
                      initial_screen=True, method=2, screen_frq=2,
                      tol_first_order=1e-3, dual_gap_inner=1e-2,
                      screen_frq_inner=5, w_init=[], maxiter_inner=1000,
                      init_iter=5, algo_method='bcd', do_screen=True):
    """ Solve Majorization Minimization with inner screening and propagation"""
    
    # init_iter : number of MM iteration with loose duality gap
    # at present we have ignore this heuristic
    n_features = X.shape[1]
    if w_init == []:
        w_mm = np.full(n_features, 0.)
    else:
        w_mm = w_init
    if type(lbd) is not np.array:
        lbd = np.ones(X.shape[1]) * lbd
    opt = False
    normXk = np.linalg.norm(X, axis=0)

    # Initialize while loop
    i = 0
    screened = np.zeros(n_features)
    screened_val = np.zeros(n_features)
    lbd_ref = np.zeros(n_features)

    while i < maxiter and not opt:
        lbdaux = lbd * approx(w_mm, theta)
        gap = 0
        if i < init_iter:
            dual_gap = dual_gap_inner
        else:
            dual_gap = dual_gap_inner
        # Transferring screening:
        if method == 2 and do_screen:
            if ((i % screen_frq) == 0):
                # computing exact screening values
                gap, rho, _, nu = compute_duality_gap_lasso(X, y, w_mm,
                                                            lbdaux)
                for k in range(n_features):
                        bound = normXk[k] * np.sqrt(2 * gap)
                        rhotxk = np.sum(nu * X[:, k])
                        screened_val[k] = np.abs(rhotxk) + bound
                        screened[k] = screened_val[k] < lbdaux[k]
                        lbd_ref[k] = lbdaux[k]
                        if screened[k]:
                            w_mm[k] = 0
                gap_old = gap
                nu_old = nu
            elif i > 1:
                # Update equation of screening values for each variable
                # using screening propagation equation
                gap, rho, _, nu = compute_duality_gap_lasso(X, y, w_mm,
                                                            lbdaux)
                bound_rho = np.linalg.norm(nu - nu_old)
                bound_gap = np.abs(gap - gap_old)
                screened = (screened_val + normXk * (bound_rho + np.sqrt(2 * bound_gap))) < lbdaux
                nu_old = nu
                gap_old = gap

        w_mm, hist_screen, out_screen, residual, correl = weighted_prox_lasso_bcd_screening(X, y, lbdaux,
                                    winit=w_mm, screen_init=screened, w_0=w_mm,
                                    dual_gap_tol=dual_gap,
                                    screen_frq=screen_frq_inner,
                                    nbitermax=maxiter_inner)
        if approx == approx_lsp:
            opt = check_opt_logsum(X, y, w_mm, lbd, theta, tol_first_order)
        i += 1
    return w_mm

#   Code used for inserting probe


def MMLasso_screening_monitoring(X, y, lbd, theta, approx=approx_lsp,
                                 maxiter=1000, initial_screen=True, method=2,
                                 screen_frq=2, tol_first_order=1e-3,
                                 dual_gap_inner=1e-2, screen_frq_inner=5,
                                 w_init=[], maxiter_inner=1000, init_iter=5,
                                 algo_method='bcd', do_screen=True):

    """ Solve Majorization Minimization with inner screening and propagation"""

    # init_iter : number of MM iteration with loose duality gap
    n_features = X.shape[1]
    if w_init == []:
        w_mm = np.full(n_features, 0.)
    else:
        w_mm = w_init
    if type(lbd) is not np.array:
        lbd = np.ones(X.shape[1]) * lbd
    opt = False
    normXk = np.linalg.norm(X, axis=0)

    # Initialize while loop
    i = 0
    screened = np.zeros(n_features)
    screened_val = np.zeros(n_features)
    lbd_ref = np.zeros(n_features)
    nb_pre_screen = np.zeros(maxiter)
    nb_post_screen = np.zeros(maxiter)
    while i < maxiter and not opt:
        lbdaux = lbd * approx(w_mm, theta)
        gap = 0
        if i < init_iter:
            dual_gap = dual_gap_inner
        else:
            dual_gap = dual_gap_inner
        # Transferring screening:
        if method == 2 and do_screen:
            if ((i % screen_frq) == 0):
                # computing exact screening values
                gap, rho, _, nu = compute_duality_gap_lasso(X, y, w_mm,
                                                            lbdaux)
                for k in range(n_features):
                        bound = normXk[k] * np.sqrt(2 * gap)
                        rhotxk = np.sum(nu * X[:, k])
                        screened_val[k] = np.abs(rhotxk) + bound
                        screened[k] = screened_val[k] < lbdaux[k]
                        lbd_ref[k] = lbdaux[k]
                        if screened[k]:
                            w_mm[k] = 0
                gap_old = gap
                nu_old = nu
            elif i > 1:
                # Update equation of screening values for each variable
                gap, rho, _, nu = compute_duality_gap_lasso(X, y, w_mm,
                                                            lbdaux)
                bound_rho = np.linalg.norm(nu - nu_old)
                bound_gap = np.abs(gap - gap_old)
                screened = (screened_val + normXk * (bound_rho + np.sqrt(2 * bound_gap))) < lbdaux
                nu_old = nu
                gap_old = gap
        nb_pre_screen[i] = np.sum(screened)
        w_mm, hist_screen, out_screen, residual, correl = weighted_prox_lasso_bcd_screening(X, y, lbdaux,
                                    winit=w_mm, screen_init=screened, w_0=w_mm,
                                    dual_gap_tol=dual_gap,
                                    screen_frq=screen_frq_inner,
                                    nbitermax=maxiter_inner)
        nb_post_screen[i] = np.sum(out_screen)
        if i==0:
            print('pre-screened \t post-screened')
        print(nb_pre_screen[i],'\t\t', nb_post_screen[i])

        if approx == approx_lsp:
            opt = check_opt_logsum(X, y, w_mm, lbd, theta, tol_first_order)
        i += 1
    return w_mm, nb_pre_screen, nb_post_screen


# Main part:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_samples = 50
    n_features = 100
    n_informative = 5
    eps = 1e-2
    sigma_bruit = 2
    np.random.seed(42)
    X, y, wopt = generate_random_gaussian(n_samples, n_features, n_informative,
                                          sigma_bruit)
    ind_wopt = np.where(abs(wopt) > 0)

    # Log-Sum
    theta = 1
    lbd = 20
    tol = 1e-5
    if 1:
        print('-------------------------------------------------------------- ')
        print('|   comparing algorithms for LSP Lasso on a single parameter   |')
        print('----------------------------------------------------------------')

        w_lsp, coutlsp = GIST(X, y, lbd, theta, reg_lsp, prox_lsp, tol=tol)

        w_bcd = BCD_noncvxlasso_lsp(X, y, lbd, theta,max_iter=3000,tol=tol)

        w_mmlsp_screened = MMLasso_screening(X, y, lbd, theta,
                                             approx=approx_lsp,
                                             initial_screen=True, method=2,
                                             screen_frq=1,
                                             tol_first_order=tol)
        tic = time()


        print("diff GIST ncxCD  : {:2.3e}".format(np.max(abs(w_lsp - w_bcd))))
        print("diff GIST MM Screen  : {:2.3e}".format(np.max(abs(w_lsp - w_mmlsp_screened))))

        print("Opt GIST : {}".format(check_opt_logsum(X, y, w_lsp, lbd, theta, tol=tol)))
        print("Opt ncxCD : {}".format(check_opt_logsum(X, y, w_bcd, lbd, theta, tol=tol)))
        print("Opt MM screen:{}".format(check_opt_logsum(X, y, w_mmlsp_screened, lbd, theta)))

        plt.plot(w_bcd), plt.plot(w_lsp), plt.plot(wopt)

    if 1:
        print('-----------------------------------------------------')
        print('|   Monitoring screening                             |')
        print('-----------------------------------------------------')

        w_mmlsp_screened = MMLasso_screening_monitoring(X, y, lbd, theta,
                                             approx=approx_lsp,
                                             initial_screen=True, method=2,
                                             screen_frq=3,
                                             tol_first_order=tol)