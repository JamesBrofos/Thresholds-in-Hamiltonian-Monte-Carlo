from typing import Callable, Tuple

import numpy as np

from .logistic import log_posterior_helper, grad_log_posterior_helper, metric_helper, grad_metric_helper


base_quantities = lambda q: (q[:-1], q[-1], np.exp(q[-1]))

def forward_transform(q):
    beta, ialpha = q[:-1], q[-1]
    log_ialpha = np.log(ialpha)
    qt = np.hstack((beta, log_ialpha))
    ildj = ialpha
    return qt, ildj

def inverse_transform(qt):
    beta, log_ialpha = qt[:-1], qt[-1]
    ialpha = np.exp(log_ialpha)
    q = np.hstack((beta, ialpha))
    fldj = -ialpha
    return q, fldj

def grad_hierarchical_helper(lin, beta, x, y, ialpha, lam, Haa):
    grad_log_ialpha = 0.5 + Haa
    grad_beta = grad_log_posterior_helper(lin, beta, x, y, ialpha)
    glp = np.hstack((grad_beta, grad_log_ialpha))
    return glp

def hessian_helper(lin, beta, x, ialpha, lam, Haa):
    Hbb = -metric_helper(lin, x, ialpha)
    Hba = -beta * ialpha
    H = np.concatenate((Hbb, Hba[..., np.newaxis]), axis=-1)
    H = np.concatenate((H, np.hstack((Hba, Haa))[np.newaxis]), axis=0)
    return H

def grad_hessian_helper(lin, beta, x, ialpha, lam, Haa):
    num_dims = len(beta)
    dHbbb = -grad_metric_helper(lin, x)
    dHbba = -np.eye(num_dims)*ialpha
    dHbaa = -ialpha*beta
    dHaaa = Haa
    da = np.concatenate((dHbbb, dHbba[..., np.newaxis]), axis=-1)
    mm = np.hstack((dHbba, dHbaa[..., np.newaxis]))
    l = np.hstack((dHbaa, dHaaa))
    rr = np.vstack((mm, l))
    db = np.concatenate((da, mm[:, np.newaxis]), axis=1)
    dH = np.concatenate((db, rr[np.newaxis]))
    return dH

def hierarchical_posterior_factory(x: np.ndarray, y: np.ndarray, lam: float) -> Tuple[Callable]:
    """Produces functions for interacting with the hierarchical Bayesian logistic
    regression wherein a exponential prior is placed over the coefficient
    precision parameter.

    Args:
        x: Covariates of the logistic regression.
        y: Binary targets of the logistic regression.
        lam: Rate parameter for the exponential prior on the precision.

    Returns:
        log_posterior: Function to compute the log-posterior.
        metric: Function to compute the Fisher information metric.
        euclidean_auxiliaries: Function to compute the log-posterior and its
            gradient.
        riemannian_auxiliaries: Function to compute the log-posterior, the
            gradient of the log-posterior, the Fisher information metric, and the
            derivatives of the Fisher information metric.

    """
    def log_posterior(q):
        beta, log_ialpha, ialpha = base_quantities(q)
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, ialpha) + 0.5*np.log(ialpha) - lam*ialpha
        return lp

    def hessian(q):
        beta, log_ialpha, ialpha = base_quantities(q)
        lin = x@beta
        Haa = - 0.5*np.dot(beta, beta)*ialpha - lam*ialpha
        H = hessian_helper(lin, beta, x, ialpha, lam, Haa)
        return H

    def euclidean_auxiliaries(q):
        beta, log_ialpha, ialpha = base_quantities(q)
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, ialpha) + 0.5*log_ialpha - lam*ialpha
        Haa = - 0.5*np.dot(beta, beta)*ialpha - lam*ialpha
        glp = grad_hierarchical_helper(lin, beta, x, y, ialpha, lam, Haa)
        return lp, glp

    def riemannian_auxiliaries(q):
        beta, log_ialpha, ialpha = base_quantities(q)
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, ialpha) + 0.5*log_ialpha - lam*ialpha
        Haa = - 0.5*np.dot(beta, beta)*ialpha - lam*ialpha
        glp = grad_hierarchical_helper(lin, beta, x, y, ialpha, lam, Haa)
        H = hessian_helper(lin, beta, x, ialpha, lam, Haa)
        dH = grad_hessian_helper(lin, beta, x, ialpha, lam, Haa)
        return lp, glp, H, dH

    return log_posterior, hessian, euclidean_auxiliaries, riemannian_auxiliaries
