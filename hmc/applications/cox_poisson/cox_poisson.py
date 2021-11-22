from typing import Callable, Tuple

import numpy as np

from scipy.spatial.distance import cdist

from hmc.linalg import solve_psd
import hmc.applications.cox_poisson.prior as prior


def forward_transform(q: np.ndarray):
    """Transform parameters from their constrained representation to their
    unconstrained representation.

    Args:
        q: The constrained parameter representation.

    Returns:
        qt: The unconstrained parameter representation.
        ildj: The logarithm of the Jacobian determinant of the inverse
            transformation.

    """
    sigmasq, beta = q
    phis, phib = np.log(sigmasq), np.log(beta)
    qt = np.array([phis, phib])
    ildj = phis + phib
    return qt, ildj

def inverse_transform(qt: np.ndarray):
    """Transform parameters from their unconstrained representation to their
    constrained representation.

    Args:
        qt: The unconstrained parameter representation.

    Returns:
        q: The constrained parameter representation.

    """
    phis, phib = qt
    sigmasq, beta = np.exp(phis), np.exp(phib)
    q = np.array([sigmasq, beta])
    fldj = -(phis + phib)
    return q, fldj

def generate_data(num_grid: int, mu: float, beta: float=1.0 / 33, sigmasq: float=1.91) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data from the log-Gaussian Cox-Poisson process.

    Args:
        num_grid: The number of grid elements for the spatial data.
        mu: Mean value of the Gaussian process.
        sigmasq: Amplitude of the Gaussian process kernel.
        beta: Length scale of the Gaussian process kernel.

    Returns:
        dist: Matrix of pairwise distances.
        x: The Gaussian process.
        y: Count observations from the Poisson process.

    """
    num_grid_sq = np.square(num_grid)
    lin = np.linspace(0.0, 1.0, num_grid)
    [I, J] = np.meshgrid(lin, lin)
    grid = np.stack((I.ravel(), J.ravel())).T
    dist = cdist(grid, grid) / num_grid
    K = sigmasq * np.exp(-dist / beta)
    L = np.linalg.cholesky(K)
    x = L@np.random.normal(size=(num_grid_sq, )) + mu
    m = 1.0 / num_grid_sq
    e = m * np.exp(x)
    y = np.random.poisson(e)
    return dist, x, y


def gaussian_posterior_factory(dist: np.ndarray, mu: float, sigmasq: float, beta: float, y: np.ndarray) -> Tuple[Callable]:
    """Factory to produce functions for computing the log-posterior, the gradient
    of the log-posterior and the Fisher information metric of the log-Gaussian
    Cox-Poisson process given values of `sigma` and `beta`.

    Args:
        dist: Matrix of pairwise distances.
        mu: Mean value of the Gaussian process.
        sigmasq: Amplitude of the Gaussian process kernel.
        beta: Length scale of the Gaussian process kernel.
        y: Count observations from the Poisson process.

    Returns:
        euclidean_auxiliaries: Function to compute the log-posterior and the
            gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.

    """
    num_grid_sq = dist.shape[0]
    m = 1.0 / num_grid_sq
    K = sigmasq * np.exp(-dist / beta)
    iK = solve_psd(K)
    # I think there should be a factor of one-half multiplying the diagonal
    # because of the expectation of the exponential of a normal random
    # variable. This differs from the RMHMC paper.
    Lambda = m * np.exp(mu + 0.5 * np.diag(K))
    G = iK.copy()
    G[np.diag_indices(G.shape[0])] += Lambda

    def euclidean_auxiliaries(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the log-posterior and the gradient of the log-posterior with
        respect to the Gaussian process given fixed values for `sigma` and
        `beta`.

        Args:
            x: The Gaussian process.

        Returns:
            lp: The log-posterior.
            glp: The gradient of the log-posterior.

        """
        o = x - mu
        e = m * np.exp(x)
        iKo = iK@o
        lp = np.sum(y*x - e) - 0.5*o.dot(iKo)
        glp = y - e - iKo
        return lp, glp

    def metric() -> np.ndarray:
        """The Fisher information metric for the log-Gaussian Cox-Poisson process. The
        Fisher information is constant with respect to the Gaussian process for
        fixed values of `sigma` and `beta`.

        Returns:
            G: The Fisher information metric.

        """
        return G

    return euclidean_auxiliaries, metric

def kernel(sigmasq: float, beta: float, dist_div_beta: np.ndarray) -> np.ndarray:
    """The smooth kernel defining spatial correlation.

    Args:
        phis: Reparameterized aplitude of the Gaussian process kernel.
        phib: Reparameterized length scale of the Gaussian process kernel.

    Returns:
        K: The kernel.

    """
    K = sigmasq * np.exp(-dist_div_beta)
    return K

def grad_kernel(K: np.ndarray, dist_div_beta: np.ndarray) -> Tuple[np.ndarray]:
    """The Hessian of the smooth kernel defining spatial correlation with respect
    to the reparameterized model parameters.

    Args:
        K: The Gaussian process covariance.
        dist_div_beta: The pairwise distances divided by the length scale

    Returns:
        dKphis: The derivative of the covariance with respect to the
            reparameterized amplitude.
        dKphib: The derivative of the covariance with respect to the
            reparameterized length scale.

    """
    dKphis = K
    dKphib = K * dist_div_beta
    return dKphis, dKphib

def hess_kernel(K: np.ndarray, dKphib: np.ndarray, dist_div_beta: np.ndarray) -> Tuple[np.ndarray]:
    """The Hessian of the smooth kernel defining spatial correlation with respect
    to the reparameterized model parameters.

    Args:
        K: The Gaussian process covariance.
        dKphib: The derivative of the covariance with respect to the
            reparameterized length scale.
        dist_div_beta: The pairwise distances divided by the length scale

    Returns:
        ddKphis: The hessian of the kernel with respect to the reparameterized
            amplitude.
        ddKphib: The hessian of the kernel with respect to the reparameterized
            length scale.
        ddKdsdb: The hessian of the kernel with respect to the reparameterized
            length scale and the reparameterized amplitude.

    """
    ddKphis = K
    ddKdsdb = dKphib
    ddKphib = K * np.square(dist_div_beta) - ddKdsdb
    return ddKphis, ddKphib, ddKdsdb

def hyperparameter_log_posterior(sigmasq: float, beta: float, oiKo: np.ndarray, L: np.ndarray) -> float:
    logdet = 2*np.sum(np.log(np.diag(L)))
    ll = -0.5*logdet - 0.5*oiKo
    pr = prior.log_prior(sigmasq, beta)
    lp = ll + pr
    return lp

def hyperparameter_grad_log_posterior(
        phis: float,
        phib: float,
        sigmasq: float,
        beta: float,
        K: np.ndarray,
        iK: np.ndarray,
        iKo: np.ndarray,
        oiKo: np.ndarray,
        dKphis: np.ndarray,
        dKphib: np.ndarray,
        iKdKphib: np.ndarray
) -> np.ndarray:
    dphis, dphib = prior.grad_log_prior(phis, phib)
    dphis += -0.5*K.shape[0] + 0.5*oiKo
    dphib += -0.5*np.trace(iKdKphib) + 0.5*iKo@dKphib@iKo
    return np.array([dphis, dphib])

def hyperparameter_metric(
        phis: float,
        phib: float,
        K: np.ndarray,
        iKdKphib: np.ndarray
) -> np.ndarray:
    a = K.shape[0]
    d = np.trace(iKdKphib)
    c = np.trace(iKdKphib@iKdKphib)
    G = 0.5 * np.array([[a, d], [d, c]])
    H = prior.hess_log_prior(phis, phib)
    F = G - H
    return F

def hyperparameter_grad_metric(
        phis: float,
        phib: float,
        K: np.ndarray,
        iK: np.ndarray,
        dKphis: np.ndarray,
        dKphib: np.ndarray,
        iKdKphib: np.ndarray,
        dist_div_beta: np.ndarray,
):
    dH = prior.grad_hess_log_prior(phis, phib)
    ddKphis, ddKphib, ddKdsdb = hess_kernel(K, dKphib, dist_div_beta)
    bb, sb = np.hsplit(iK@np.hstack((ddKphib, ddKdsdb)), 2)
    b = iKdKphib
    b_b = b@b
    s_b_b = b_b
    # I think that the partial derivative of the metric with respect to the
    # amplitude vanishes because the amplitude cancels in the metric.
    dGs = np.array([[0.0, 0.0], [0.0, 0.0]])
    od = 0.5*np.trace(-s_b_b + bb)
    b_bb = b@bb
    dGb = np.array([[0.0, od], [od, np.trace(-b@b_b + b_bb)]])
    dG = np.array([dGs, dGb]).swapaxes(0, -1)
    return dG - dH

def base_quantities(qt, dist, o):
    phis, phib = qt
    sigmasq = np.exp(phis)
    beta = np.exp(phib)
    dist_div_beta = dist / beta
    K = kernel(sigmasq, beta, dist_div_beta)
    iK, L = solve_psd(K, return_chol=True)
    iKo = iK@o
    oiKo = o@iKo
    return phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo

def hyperparameter_posterior_factory(dist: np.ndarray, mu: float, x: np.ndarray, y: np.ndarray) -> Tuple[Callable]:
    """Factory to produce the log-posterior, the gradient of the log-posterior, the
    Fisher information metric and the gradient of the Fisher information metric
    for the log-Gaussian Cox-Poisson process given the underlying Gaussian
    process.

    Args:
        dist: Matrix of pairwise distances.
        mu: Mean value of the Gaussian process.
        x: The Gaussian process.
        y: Count observations from the Poisson process.

    Returns:

    """
    o = x - mu

    def log_posterior(qt: np.ndarray) -> float:
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, dist, o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        return lp

    def euclidean_auxiliaries(qt: np.ndarray) -> Tuple[np.ndarray]:
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, dist, o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        glp = hyperparameter_grad_log_posterior(phis, phib, sigmasq, beta, K, iK, iKo, oiKo, dKphis, dKphib, iKdKphib)
        return lp, glp

    def metric(qt: np.ndarray) -> np.ndarray:
        phis, phib = qt
        sigmasq = np.exp(phis)
        beta = np.exp(phib)
        dist_div_beta = dist / beta
        K = kernel(sigmasq, beta, dist_div_beta)
        iK, L = solve_psd(K, return_chol=True)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        return G

    def riemannian_auxiliaries(qt: np.ndarray) -> Tuple[np.ndarray]:
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, dist, o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        glp = hyperparameter_grad_log_posterior(phis, phib, sigmasq, beta, K, iK, iKo, oiKo, dKphis, dKphib, iKdKphib)
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        dG = hyperparameter_grad_metric(phis, phib, K, iK, dKphis, dKphib, iKdKphib, dist_div_beta)
        return lp, glp, G, dG

    def log_posterior_and_metric(qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, dist, o)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        iKdKphib = iK@dKphib
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        return lp, G

    return (
        log_posterior,
        metric,
        log_posterior_and_metric,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    )

