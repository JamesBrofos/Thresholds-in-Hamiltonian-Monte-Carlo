from typing import Callable, Tuple

import numpy as np

from hmc.linalg import solve_psd
from hmc.statistics import rvs


def sample(sqrtm: np.ndarray, dof: float) -> np.ndarray:
    """Draws samples from the multivariate Student-t distribution given square root
    of the covariance matrix and the degrees of freedom.

    Args:
        sqrtm: The square root of the covariance matrix.
        dof: The degrees of freedom.

    Returns:
        x: A sample from the multivariate Student-t distribution.

    """
    zero = np.zeros(len(sqrtm))
    y = rvs(sqrtm, zero)
    u = np.random.chisquare(dof)
    x = y * np.sqrt(dof / u)
    return x

def base_quantities(x: np.ndarray, iSigma: np.ndarray, dof: float) -> Tuple[np.ndarray]:
    """Computes the preconditioning effect of the inverse covariance matrix on the
    input and the inner product regularized by the degrees of freedom.

    Args:
        x: A point in Euclidean space.
        iSigma: The inverse of the covariance matrix.
        dof: The degrees of freedom.

    Returns:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.

    """
    iSigma_x = iSigma@x
    ip = 1 + x@iSigma_x / dof
    return iSigma_x, ip

def log_posterior_helper(ip: np.ndarray, dof: float, n: int) -> float:
    """Computes the log-density of the Student-t distribution given the inner
    product, the degrees of freedom, and the dimensionality of the space.

    Args:
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        lp: The log-density.

    """
    lp = -0.5*(dof + n)*np.log(ip)
    return lp

def grad_log_posterior_helper(iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes the gradient of the log-density of the Student-t distribution with
    respect to a location in Euclidean space.

    Args:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        glp: The gradient of the log-posterior.

    """
    glp = -(dof + n) / ip * iSigma_x / dof
    return glp

def hessian_helper(iSigma, iSigma_x, ip, dof, n):
    k = dof + n
    H = (
        -k / ip * iSigma / dof
        + 2*k / dof**2 / ip**2 * np.outer(iSigma_x, iSigma_x)
    )
    return H

def grad_hessian_helper(x, iSigma, iSigma_x, ip, dof, n):
    k = dof + n
    Id = np.eye(n)
    o = np.einsum('ki,j->ijk', Id, x) + np.einsum('i,kj->ijk', x, Id)
    o = np.swapaxes(o, 0, -1)
    o = np.swapaxes(iSigma@o@iSigma, 0, -1)
    dH = (
        2*k / ip**2 * np.einsum('ij,k->ijk', iSigma, iSigma_x) / dof**2
        + 2*k / dof**2 / ip**2 * o
        - 8*k / dof**3 / ip**3 * np.einsum('i,j,k->ijk', iSigma_x, iSigma_x, iSigma_x)
    )
    return dH

def metric_helper(iSigma, iSigma_x, ip, dof, n):
    k = dof + n
    G = k / ip * iSigma / dof
    return G

def grad_metric_helper(x, iSigma, iSigma_x, ip, dof, n):
    k = dof + n
    dG = -2*k / ip**2 * np.einsum('ij,k->ijk', iSigma, iSigma_x) / dof**2
    return dG

def posterior_factory(Sigma: np.ndarray, dof: float) -> Tuple[Callable]:
    """Constructs the posterior distribution corresponding to a multivariate
    Student-t distribution given the covariance matrix and the degrees of
    freedom.

    Args:
        Sigma: The covariance matrix of the Student-t distribution.
        dof: The degrees of freedom of the Student-t distribution.

    """
    iSigma, L = solve_psd(Sigma, return_chol=True)
    n = len(Sigma)
    def log_posterior(x):
        _, ip = base_quantities(x, iSigma, dof)
        lp = log_posterior_helper(ip, dof, n)
        return lp

    def grad_log_posterior(x):
        iSigma_x, ip = base_quantities(x, iSigma, dof)
        glp = grad_log_posterior_helper(iSigma_x, ip, dof, n)
        return glp

    def euclidean_auxiliaries(x):
        iSigma_x, ip = base_quantities(x, iSigma, dof)
        lp = log_posterior_helper(ip, dof, n)
        glp = grad_log_posterior_helper(iSigma_x, ip, dof, n)
        return lp, glp

    def metric(x):
        iSigma_x, ip = base_quantities(x, iSigma, dof)
        G = metric_helper(iSigma, iSigma_x, ip, dof, n)
        return G

    def riemannian_auxiliaries(x):
        iSigma_x, ip = base_quantities(x, iSigma, dof)
        lp = log_posterior_helper(ip, dof, n)
        glp = grad_log_posterior_helper(iSigma_x, ip, dof, n)
        G = metric_helper(iSigma, iSigma_x, ip, dof, n)
        dG = grad_metric_helper(x, iSigma, iSigma_x, ip, dof, n)
        return lp, glp, G, dG

    def log_posterior_and_metric(x):
        iSigma_x, ip = base_quantities(x, iSigma, dof)
        lp = log_posterior_helper(ip, dof, n)
        G = metric_helper(iSigma, iSigma_x, ip, dof, n)
        return lp, G

    return (
        log_posterior,
        metric,
        log_posterior_and_metric,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    )
