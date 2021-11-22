from typing import Tuple

import numpy as np


def inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The log-density function of the inverse chi-squared distribution up to a
    constant factor.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The log-density of the inverse chi-squared distribution.

    """
    hv = 0.5*v
    a = hv * (np.log(t) + np.log(hv))
    # Should include a log-Gamma function term but this requires using SciPy.
    # Because this is just a constant factor, it can be safely ignored for the
    # purposes of HMC.
    # >>> import scipy.special as spsp
    # >>> b = -spsp.gammaln(hv)
    b = 0.0
    c = -v*t / (2.0 * x)
    d = -(1.0 + hv) * np.log(x)
    return a + b + c + d

def grad_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The gradient of the log-density function of the inverse chi-squared
    distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The gradient of the log-density of the inverse chi-squared
            distribution.

    """
    hv = 0.5*v
    return -(1.0 + hv) / x + 0.5 * v*t / np.square(x)

def hess_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The hessian of the log-density function of the inverse chi-squared
    distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The hessian of the log-density of the inverse chi-squared
            distribution.

    """
    hv = 0.5*v
    return (1.0 + hv) / np.square(x) - v*t / np.power(x, 3.0)

def grad_hess_inv_chisq_logpdf(x: float, v: float, t: float) -> float:
    """The third-order derivative of the log-density function of the inverse
    chi-squared distribution.

    Args:
        x: The location at which to evaluate the density of the inverse
            chi-squared distribution.
        v: Degrees of freedom of inverse chi-squared distribution.
        t: Scale factor of the inverse chi-squared distribution.

    Returns:
        out: The third-order derivative of the log-density of the inverse
            chi-squared distribution.

    """
    hv = 0.5*v
    return -2.0 * (1.0 + hv) / np.power(x, 3.0) + 3.0 * v*t / np.power(x, 4.0)

def beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """The log-density function of the Beta distribution up to a constant
    factor.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        lp: The log-density of the Beta distribution.

    """
    lp = (alpha-1.0)*np.log(p) + (beta-1.0)*np.log(1-p)
    return lp

def grad_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Gradient of the log-density function of the Beta distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The gradient of the log-density of the Beta distribution.

    """
    return (alpha - 1.0) / p - (beta - 1.0) / (1.0 - p)

def hess_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Hessian of the log-density function of the Beta distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The hessian of the log-density of the Beta distribution.

    """
    return (1.0 - alpha) / np.square(p) + (1.0 - beta) / np.square(p - 1.0)

def grad_hess_beta_logpdf(p: float, alpha: float, beta: float) -> float:
    """Third-order derivative of the log-density function of the Beta
    distribution.

    Args:
        p: A point on the unit interval at which to evaluate the density.
        alpha: Beta distribution 'success' parameter.
        beta: Beta distribution 'failure' parameter.

    Returns:
        out: The third-order derivative of the log-density of the Beta
            distribution.

    """
    a = 2 * (alpha - 1.0) / np.power(p, 3.0)
    b = 2 * (beta - 1.0) / np.power(p - 1.0, 3.0)
    return a + b

def log_prior(sigma: float, phi: float, beta: float) -> float:
    """The log-prior density for the stochastic volatility model. The prior
    distribution is as follows (notice that the beta prior is improper):

    sigma ~ InvChiSquare(10.0, 0.05)
    (phi + 1) / 2 ~ Beta(20, 1.5)
    p(beta) = 1 / beta

    Args:
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        out: The log-density of the prior.

    """
    sigmasq = np.square(sigma)
    lbeta = -np.log(beta)
    lphi = beta_logpdf(0.5*(phi + 1.0), 20.0, 1.5)
    lsigmasq = inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    return lbeta + lphi + lsigmasq

def grad_log_prior(gamma: float, alpha: float, beta: float) -> Tuple[float]:
    """The gradient of the log-density of the prior for the stochastic volatility
    model. We reparameterize `sigma` and `phi` to respect parameter
    constraints.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        glp: The gradient of the prior log-density.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    dbeta = -1.0 / beta
    dgamma = 2.0 * sigmasq * grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    dalpha = 0.5 * (1.0 - phisq) * grad_beta_logpdf(0.5*(phi + 1.0), 20.0, 1.5)
    glp = np.array((dgamma, dalpha, dbeta))
    return glp

def hess_log_prior(gamma: float, alpha: float, beta: float) -> np.ndarray:
    """The hessian of the log-density of the prior for the stochastic volatility
    model.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        H: The Hessian of the prior log-density.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    m = 0.5*(phi + 1.0)
    a = 4.0*sigmasq*grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 4.0*np.square(sigmasq)*hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    b = -phi*(1.0 - phisq)*grad_beta_logpdf(m, 20.0, 1.5)
    b += 0.25*np.square(1.0 - phisq)*hess_beta_logpdf(m, 20.0, 1.5)
    c = 1.0 / np.square(beta)
    H = np.array(([a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]))
    return H

def grad_hess_log_prior(gamma: float, alpha: float, beta: float) -> np.ndarray:
    """The tensor of higher-order derivatives of the log-density of the prior for
    the stochastic volatility model.

    Args:
        gamma: Parameter of the stochastic volatility model.
        alpha: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        dH: The tensor of higher-order derivatives of the prior log-density.

    """
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    sigmaquad = np.square(sigmasq)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    m = 0.5*(phi + 1.0)
    r = 1 - phisq
    h = hess_beta_logpdf(m, 20.0, 1.5)
    dHbeta = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -2.0 / np.power(beta, 3.0)]
    ])
    ghx = grad_hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a = 8.0*sigmasq*grad_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 24.0*sigmaquad*hess_inv_chisq_logpdf(sigmasq, 10.0, 0.05)
    a += 8.0*sigmaquad*sigmasq*ghx
    dHgamma = np.array([
        [a, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    ])
    b = r*(3.0*phisq - 1.0)*grad_beta_logpdf(m, 20.0, 1.5)
    b -= 0.5*phi*np.square(r)*h
    b -= phi*np.square(r)*h
    b += np.power(r, 3.0)*grad_hess_beta_logpdf(m, 20.0, 1.5) / 8.0
    dHalpha = np.array([
        [0.0, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, 0.0]
    ])
    return np.stack((dHgamma, dHalpha, dHbeta), axis=0)
