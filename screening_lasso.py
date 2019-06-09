# -*- coding: utf-8 -*-
"""Trying screening code on sparse methods."""
# @author: alain


from time import process_time as time
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse import csr_matrix
import numpy as np


def generate_random_gaussian(n_samples=50, n_features=100, n_informative=2,
                             sigma_bruit=0, scal=2):

    """ build the toy problem"""
    
    X = np.random.randn(n_samples, n_features) * scal
    ind = np.random.permutation(n_features)
    wopt = np.zeros(n_features)
    aux = np.random.randn(n_informative)
    wopt[ind[0:n_informative]] = aux + 0.1 * np.sign(aux)
    y = X.dot(wopt) + np.random.randn(n_samples) * sigma_bruit

    return X, y, wopt

#
def prox_l1(w, thresh):
    """Soft-thresholding function."""
    absw = np.abs(w)
    return np.where((absw - thresh) > 0, (absw - thresh) * np.sign(w), 0)

#
def compute_duality_gap_lasso(X, y, w, lbd):
        """Compute duality gap for Lasso.
        """
        # part of the code is designed for sparse matrix.
        if type(lbd) is not np.array:
            lbd = np.ones(X.shape[1]) * lbd
        rho = y - X.dot(w)

        
        if not isspmatrix(X):
            normrho2 = np.linalg.norm(rho, axis=0, ord=2)**2
            primalcost = 0.5 * normrho2 + np.sum(np.abs(w) * lbd)
        else:
            normrho2 = spnorm(rho, axis=0)**2
            primalcost = 0.5 * normrho2 + (abs(w*lbd).sum())

        correl = X.T.dot(rho)


        # nu is a feasible approximation of nu
        max_viol = np.max(np.abs(correl) / lbd)

        if max_viol > 1:
            nu = rho / max_viol
        else:
            nu = rho
        if not isspmatrix(X):
            dualcost = - 0.5 * np.sum(nu * nu) + np.sum(nu * y)
        else:
            dualcost = -0.5 * spnorm(nu)**2 + (nu.multiply(y)).sum()
        gap = primalcost - dualcost
        return gap, rho, correl, nu


def compute_duality_gap_prox_lasso(X, y, w, lbd, w_0, alpha):
        """Compute duality gap for prox_Lasso."""
        if type(lbd) is not np.array:
            lbd = np.ones(X.shape[1]) * lbd

        rho = y - X.dot(w)

        if not isspmatrix(X):
            normrho2 = np.linalg.norm(rho, axis=0, ord=2)**2
            primalcost = 0.5 * normrho2 + np.sum(np.abs(w) * lbd) + \
                0.5 / alpha * np.sum((w - w_0)**2)
        else:
            normrho2 = spnorm(rho, axis=0)**2
            primalcost = 0.5 * normrho2 + (abs(w*lbd).sum())
        v = (w - w_0) / alpha
        correl = X.T.dot(rho)

        max_viol = np.max(np.abs(correl - v) / lbd)

        if max_viol > 1:
            nu = rho / max_viol
            v = v / max_viol
        else:
            nu = rho
        if not isspmatrix(X):
            dualcost = - 0.5 * np.sum(nu * nu) + np.sum(nu * y) - \
                v.T.dot(w_0) - alpha / 2 * np.sum(v**2)
        else:
            dualcost = -0.5 * spnorm(nu)**2 + (nu.multiply(y)).sum() - \
                v.T.dot(w_0) - alpha / 2 * np.sum(v**2)
        gap = primalcost - dualcost
        return gap, rho, correl, nu, v


def check_opt_prox_lasso(X, y, w, lbd, w_0, alpha, tol=1e-4):
    """Check the optimality of a weighted Lasso problem."""
    tol_zero = 1e-5  # defined in hard. not to confused with tol on constraints
    if type(lbd) is not np.array:
        lbd = np.ones(X.shape[1]) * lbd
    residue = y - X.dot(w)
    v = (w - w_0) / alpha
    correl = -(X.T.dot(residue) )
    idx = np.where(np.abs(w) > tol)
    optimal_positive = np.all(np.abs(np.abs(correl[idx] +
                                     v[idx]) - lbd[idx]) < tol)
    idx_o = np.where(np.abs(w) < tol_zero)
    optimal_zero = np.all(np.abs(correl[idx_o] + v[idx_o]) < lbd[idx_o])
    optimality = optimal_positive and optimal_zero
    return optimality, correl

def weighted_lasso_bcd(X, y, lbd, nbitermax=100, dual_gap_tol=1e-6,
                       winit=np.array([])):
    """
    solve min_w 0.5 *|| y - Xw ||^2 +  \sum_i \lbda_i |w_i|

    using coordinatewise descent 

    """
    
    
    n_features = X.shape[1]
    normX = np.linalg.norm(X, axis=0, ord=2)
    if type(lbd) is not np.array:
        lbd = np.ones(X.shape[1]) * lbd
    if len(winit) == 0:
        w = np.zeros(n_features)
    else:
        w = np.copy(winit)

    i = 0
    gap = float("inf")
    while i < nbitermax and gap > dual_gap_tol:
        for k in range(n_features):
                waux = np.copy(w)
                waux[k] = 0
                s = y - X.dot(waux)
                Xts = X[:, k].T.dot(s)
                w[k] = prox_l1(Xts, lbd[k]) / (normX[k]**2)  #np.sign(Xts) * np.maximum((np.abs(Xts) - lbd), 0) / (normX[k]**2)
        gap, rho, correl, nu = compute_duality_gap_lasso(X, y, w, lbd)
        # print(i,gap)
        i += 1
    return w
#
#
def weighted_lasso_bcd_screening(X, y, lbd, nbitermax=10000,
                                 winit=np.array([]),
                                 screen_init=np.array([]),
                                 screen_frq=3, dual_gap_tol=1e-6):
    """
    solve min_w 0.5 *|| y - Xw ||^2 +  \sum_i \lbda_i |w_i|
    using coordinatewise descent + screening
    """
    
    n_features = X.shape[1]
    bool_sparse = isspmatrix(X)
    
    
    if not bool_sparse:
        normX = np.linalg.norm(X, axis=0, ord=2)
        if len(winit) == 0:
            w = np.zeros(n_features)
        else:
            w = np.copy(winit)
    else:
        normX = spnorm(X, axis=0)
        winit = csr_matrix(winit)
        if winit.nnz == 0:
            w = csr_matrix(np.zeros(n_features))
        else:
            w = winit.copy()
    
    if len(screen_init) == 0:
        screened = np.zeros(n_features)
    else:
        screened = np.copy(screen_init)
    nb_screened = np.zeros(nbitermax)
    i = 0
    gap = float("inf")
    if bool_sparse:
        w = csr_matrix(w.reshape(-1, 1))
    rho = y - X.dot(w)
    for i in range(nbitermax):
        for k in range(n_features):
            if not screened[k]:
                # computing the partial residual through update of the res
                xk = X[:, k]
                s = rho + (xk * w[k])
                Xts = xk.T.dot(s)
                if not bool_sparse:
                    # w[k] = np.sign(Xts)* np.maximum((np.abs(Xts) - lbd[k]),0)
                    wp = prox_l1(Xts, lbd[k]) / (normX[k]**2)  #np.sign(Xts) * np.maximum((np.abs(Xts) - lbd), 0) / (normX[k]**2)
                    rho = rho - xk * (wp - w[k])
                    w[k] = wp
    
                else:  # sparse
                    Xts = Xts[0, 0]
                    wp = prox_l1(Xts, lbd[k]) / (normX[k]**2)  #np.sign(Xts) * np.maximum((np.abs(Xts) - lbd), 0) / (normX[k]**2)
                    # wp = np.sign(Xts) * np.maximum((np.abs(Xts) - lbd[k]), 0) / (normX[k]**2)
                    rho = rho - xk * (csr_matrix(wp) - w[k])
    
        gap, rho, correl, nu = compute_duality_gap_lasso(X, y, w, lbd)
        if i % screen_frq == 0:
            # Updating screeening
            for k in range(n_features):
                    if not screened[k]:
                        bound = normX[k] * np.sqrt(2 * gap)
                        if not bool_sparse:
                            rhotxk = np.sum(nu * X[:, k])
                        else:
                            rhotxk = nu.multiply(X[:, k]).sum()
                        screened[k] = ((abs(rhotxk) + bound) < lbd[k])
            nb_screened[i] = (sum(screened == 1))
        else:
            nb_screened[i] = nb_screened[i - 1]
            gap = float("inf")
        i += 1
        #print('L', i, gap, 's', nb_screened[i - 1])
        if gap < dual_gap_tol:
            # print('L Sorti', gap)
            break
    
    return w, nb_screened[:i], screened, rho, correl


def weighted_prox_lasso_bcd_screening(X, y, lbd, w_0=np.array([]),alpha=1e9, nbitermax=10000,
                                      winit=np.array([]),
                                      screen_init=np.array([]), screen_frq=3,
                                      dual_gap_tol=1e-6,do_screen = True):
    """
    solve min_w 0.5 *|| y - Xw ||^2 + 1/2\alpha ||w - w_0||^2 + \sum_i \lbda_i |w_i|

    using coordinatewise descent + screening

    """
    n_features = X.shape[1]
    bool_sparse = isspmatrix(X)
    if not bool_sparse:
        normX = np.linalg.norm(X, axis=0, ord=2)
        if len(winit) == 0:
            w = np.zeros(n_features)
        else:
            w = np.copy(winit)
    else:
        normX = spnorm(X, axis=0)
        winit = csr_matrix(winit)
        if winit.nnz == 0:
            w = csr_matrix(np.zeros(n_features))
        else:
            w = winit.copy()

    if len(screen_init) == 0:
        screened = np.zeros(n_features)
    else:
        screened = np.copy(screen_init)
    
    nb_screened = np.zeros(nbitermax)
    if len(w_0) == 0:
        w_0 = np.zeros(n_features)
    
    
    i = 0
    gap = float("inf")
    if bool_sparse:
        w = csr_matrix(w.reshape(-1, 1))
    rho = y - X.dot(w)
    while i < nbitermax and gap > dual_gap_tol:
        for k in range(n_features):
            if not screened[k]:
                # Updating the variable
                # computing the partial residual through update of the residual
                xk = X[:, k]
                s = rho + (xk * w[k])
                Xts = (xk.T.dot(s) + w_0[k] / alpha)
                wp = prox_l1(Xts, lbd[k]) / (normX[k]**2 + 1 / alpha)
                rho = rho - xk * (wp - w[k])
                w[k] = wp
        gap, rho, correl, nu, v = compute_duality_gap_prox_lasso(X, y, w, lbd,
                                                                 w_0, alpha)
        if i % screen_frq == 0 and do_screen:
            # Updating screeening
            for k in range(n_features):
                    if not screened[k]:
                        bound = (normX[k] + 1 / alpha) * np.sqrt(2 * gap)
                        if not bool_sparse:
                            rhotxk = np.sum(nu * X[:, k])
                        else:
                            rhotxk = nu.multiply(X[:, k]).sum()
                        screened[k] = ((abs(rhotxk - v[k]) + bound) < lbd[k])
            nb_screened[i] = (sum(screened == 1))
        else:
            nb_screened[i] = nb_screened[i - 1]
        #print(i,gap,nb_screened[i])
        i += 1
    return w, nb_screened[:i], screened, rho, correl





if __name__ == "__main__":

    np.random.seed(42)
    n_informative = 5
    n_samples = 50
    n_features = 100
    sigma_bruit = 1
    X, y, wopt = generate_random_gaussian(n_samples, n_features, n_informative,
                                          sigma_bruit)
    lbdmax = np.max(np.abs(X.T.dot(y)))

    # switch commented lines if you want to try non-uniform lambda
    #lbd = np.maximum(0.5 + np.random.randn(n_features) * 10, 1)
    lbd = np.ones(n_features) * lbdmax / 10 # uniform lambda

    w_0 = np.random.randn(n_features)
    alpha = 10000
    tic = time()
    w_wplscreened, _, screened, residual, correl = weighted_prox_lasso_bcd_screening(X, y, lbd,w_0,alpha,nbitermax=20000)
    time_wpls = time() - tic
    optimality_wpls, correl_wpls = check_opt_prox_lasso(X, y, w_wplscreened,
                                                        lbd, w_0, alpha)

    print('inf norm error:', np.max(abs(wopt-w_wplscreened )))
    print('optimality:',optimality_wpls)