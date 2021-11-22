from typing import Callable, Tuple

import numpy as np
import scipy.linalg as spla

from hmc.statistics import rvs


def sample(mu: np.ndarray, sqrtm: np.ndarray) -> np.ndarray:
    """Samples from the Gaussian target distribution given a mean vector and a
    matrix square root of the covariance.

    Args:
        mu: The mean of the multivariate normal distribution.
        sqrtm: The matrix square root of the covariance matrix of the multivariate
            normal distribution.

    Returns:
        x: The Gaussian sample.

    """
    x = rvs(sqrtm, mu)
    return x

def posterior_factory(mu: np.ndarray, Sigma: np.ndarray) -> Tuple[Callable]:
    """Implements sampling from a multivariate normal distribution. Constructs
    functions for the log-density of the normal distribution and for the
    gradient of the log-density.

    Args:
        mu: The mean of the multivariate normal distribution.
        Sigma: The covariance matrix of the multivariate normal distribution.

    Returns:
        log_posterior: The log-density of the multivariate normal.
        grad_log_posterior: The gradient of the log-density of the multivariate
            normal distribution.
        metric: The metric for the multivariate normal.

    """
    mu = np.atleast_1d(mu)
    Sigma = np.atleast_2d(Sigma)
    n = len(mu)
    L = spla.cholesky(Sigma)
    iL = spla.solve_triangular(L, np.eye(n))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    iSigma = iL@iL.T

    # Check calculations:
    # >>> np.allclose(logdet, np.log(np.linalg.det(Sigma)))
    # >>> np.allclose(iSigma, np.linalg.inv(Sigma))

    def metric(x: np.ndarray) -> np.ndarray:
        """Use the covariance matrix as a constant metric.

        Args:
            x: The Gaussian sample.

        Returns:
            Sigma: The covariance matrix of the multivariate normal distribution.

        """
        return iSigma

    def euclidean_auxiliaries(x: np.ndarray) -> Tuple[np.ndarray]:
        """Computes the log-posterior and the gradient of the log-posterior of the
        Gaussian distribution.

        Args:
            x: The Gaussian sample.

        Returns:
            lp: The log-density.
            glp: The gradient of the log-density.

        """
        o = x - mu
        glp = -o@iSigma
        maha = np.sum(glp*o, axis=-1)
        lp = -0.5*n*np.log(2.0*np.pi) - 0.5*logdet + 0.5*maha
        return lp, glp

    return euclidean_auxiliaries, metric
