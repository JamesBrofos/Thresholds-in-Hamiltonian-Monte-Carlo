from typing import Tuple

import numpy as np


def gamma_logpdf(x: float, k: float, theta: float) -> float:
    """Log-density of the Gamma distribution up to a constant factor.

    Args:
        x: Positive number at which to evaluate the Gamma distribution.
        k: Shape parameter of the Gamma distribution.
        theta: Scale parameter of the Gamma distribution.

    Returns:
        out: The log-density of the Gamma distribution.

    """
    return (k - 1.0)*np.log(x) - x / theta

def grad_gamma_logpdf(x: float, k: float, theta: float) -> float:
    """Gradient of the log-density of the Gamma distribution.

    Args:
        x: Positive number at which to evaluate the Gamma distribution.
        k: Shape parameter of the Gamma distribution.
        theta: Scale parameter of the Gamma distribution.

    Returns:
        out: The gradient of the log-density of the Gamma distribution.

    """
    return (k - 1.0) / x - np.reciprocal(theta)

def hess_gamma_logpdf(x: float, k: float, theta: float) -> float:
    """Hessian of the log-density of the Gamma distribution.

    Args:
        x: Positive number at which to evaluate the Gamma distribution.
        k: Shape parameter of the Gamma distribution.
        theta: Scale parameter of the Gamma distribution.

    Returns:
        out: The Hessian of the log-density of the Gamma distribution.

    """
    return -(k - 1.0) / np.square(x)

def grad_hess_gamma_logpdf(x: float, k: float, theta: float) -> float:
    """Third-order derivatives of the log-density of the Gamma distribution.

    Args:
        x: Positive number at which to evaluate the Gamma distribution.
        k: Shape parameter of the Gamma distribution.
        theta: Scale parameter of the Gamma distribution.

    Returns:
        out: The third-order derivative of the log-density of the Gamma
            distribution.

    """
    return 2.0*(k - 1.0) / np.power(x, 3.0)

def log_prior(sigmasq: float, beta: float) -> float:
    """The log-prior of the log-Gaussian Cox-Poisson model.

    Args:
        sigmasq: Amplitude of the Gaussian process kernel.
        beta: Length scale of the Gaussian process kernel.

    Returns:
        lp: The log-density of the prior distribution.

    """
    lp = gamma_logpdf(beta, 2.0, 0.5)
    lp += gamma_logpdf(sigmasq, 2.0, 0.5)
    return lp

def grad_log_prior(phis: float, phib: float) -> Tuple[float]:
    """Gradient of the log-prior with respect to the reparameterized model
    parameters that are unconstrained.

    Args:
        phis: Reparameterized aplitude of the Gaussian process kernel.
        phib: Reparameterized length scale of the Gaussian process kernel.

    Returns:
        out: The gradient of the log-prior with respect to the reparameterized
            model parameters.

    """
    sigmasq = np.exp(phis)
    beta = np.exp(phib)
    dphis = grad_gamma_logpdf(sigmasq, 2.0, 0.5) * sigmasq
    dphib = grad_gamma_logpdf(beta, 2.0, 0.5) * beta
    return np.array((dphis, dphib))

def hess_log_prior(phis: float, phib: float) -> np.ndarray:
    """Compute the hessian of the log-prior with respect to the reparameterized
    model parameters.

    Args:
        phis: Reparameterized aplitude of the Gaussian process kernel.
        phib: Reparameterized length scale of the Gaussian process kernel.

    Returns:
        H: The Hessian of the log-prior with respect to the reparameterized model
            parameters.

    """
    sigmasq = np.exp(phis)
    beta = np.exp(phib)
    H = np.array([
        [grad_gamma_logpdf(sigmasq, 2.0, 0.5)*sigmasq + np.square(sigmasq)*hess_gamma_logpdf(sigmasq, 2.0, 0.5), 0.0],
        [0.0, grad_gamma_logpdf(beta, 2.0, 0.5)*beta + np.square(beta)*hess_gamma_logpdf(beta, 2.0, 0.5)]
    ])
    return H

def grad_hess_log_prior(phis: float, phib: float) -> np.ndarray:
    """Compute the third-order derivatives of the log-prior with respect to the
    reparameterized model parameters.

    Args:
        phis: Reparameterized aplitude of the Gaussian process kernel.
        phib: Reparameterized length scale of the Gaussian process kernel.

    Returns:
        dH: The third-order derivatives of the log-prior.

    """
    sigmasq = np.exp(phis)
    beta = np.exp(phib)
    dH = np.zeros((2, 2, 2))
    a = sigmasq*grad_gamma_logpdf(sigmasq, 2.0, 0.5)
    a += np.square(sigmasq)*hess_gamma_logpdf(sigmasq, 2.0, 0.5)
    a += 2.0*sigmasq*hess_gamma_logpdf(sigmasq, 2.0, 0.5)
    a += np.square(sigmasq)*grad_hess_gamma_logpdf(sigmasq, 2.0, 0.5)
    b = beta*grad_gamma_logpdf(beta, 2.0, 0.5)
    b += np.square(beta)*hess_gamma_logpdf(beta, 2.0, 0.5)
    b += 2.0*beta*hess_gamma_logpdf(beta, 2.0, 0.5)
    b += np.square(beta)*grad_hess_gamma_logpdf(beta, 2.0, 0.5)
    dH = np.array([
        [[a, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, b]]
    ])
    return dH
