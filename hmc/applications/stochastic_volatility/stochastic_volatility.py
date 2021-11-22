from typing import Callable, Tuple

import numpy as np

import hmc.applications.stochastic_volatility.prior as prior


def forward_transform(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters frm their constrained representation to their
    unconstrained representation.

    Args:
        q: The constrained parameter representation.

    Returns:
        qt: The unconstrained parameter representation.
        ildj: The logarithm of the Jacobian determinant of the inverse
            transformation.

    """
    sigma, phi, beta = q
    gamma, alpha = np.log(sigma), np.arctanh(phi)
    qt = np.array([gamma, alpha, beta])
    ildj = gamma + np.log(1.0 - np.square(phi))
    return qt, ildj

def inverse_transform(qt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters from their unconstrained representation to their
    constrained representation.

    Args:
        qt: The unconstrained parameter representation.

    Returns:
        q: The constrained parameter representation.
        fldj: The logarithm of the Jacobian determinant of the forward
            transformation.

    """
    gamma, alpha, beta = qt
    sigma, phi = np.exp(gamma), np.tanh(alpha)
    q = np.array([sigma, phi, beta])
    fldj = -(gamma + np.log(1.0 - np.square(phi)))
    return q, fldj

def generate_data(T: int, sigma: float=0.15, phi: float=0.98, beta: float=0.65) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples from the stochastic volatility model.

    Args:
        T: Number of subsequent time points at which to generate an observation
            of the stochastic volatility model.
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.

    Returns:
        x: The stochastic volatilities.
        y: Observations from the stochastic volatility model.

    """
    x = np.random.normal(size=(T, ))
    x[0] = np.random.normal(0.0, sigma / np.sqrt(1.0 - np.square(phi)))
    for t in range(1, T):
        eta = np.random.normal(0.0, sigma)
        x[t] = phi * x[t-1] + eta

    eps = np.random.normal(size=x.shape)
    y = eps * beta * np.exp(0.5 * x)
    return x, y

def volatility_log_posterior(x: np.ndarray,
                             omega: np.ndarray,
                             omegasq: np.ndarray,
                             sigma: float,
                             sigmasq: float,
                             phi: float,
                             phisq: float,
                             beta: float,
                             T: int,
                             S: int,
                             ysq: np.ndarray) -> float:
    ly = -0.5*np.sum(ysq / omegasq) - np.sum(np.log(omega)) - 0.5*T*np.log(2*np.pi)
    sigma_xo = sigma / np.sqrt(1.0 - phisq)
    lxo = -0.5*np.square(x[0] / sigma_xo) - np.log(sigma_xo) - 0.5*np.log(2*np.pi)
    lx = -0.5*np.sum(np.square((x[1:] - phi*x[:-1]) / sigma)) - S*np.log(sigma) - 0.5*S*np.log(2*np.pi)
    lp = ly + lx + lxo
    return lp

def volatility_grad_log_posterior(x: np.ndarray,
                                  omega: np.ndarray,
                                  omegasq: np.ndarray,
                                  sigma: float,
                                  sigmasq: float,
                                  phi: float,
                                  phisq: float,
                                  beta: float,
                                  T: int,
                                  S: int,
                                  ysq: np.ndarray) -> np.ndarray:
    s = 0.5 * (ysq / omegasq - 1.0)
    do = (x[0] - phi * x[1]) / sigmasq
    dn = (x[-1] - phi * x[-2]) / sigmasq
    w = (x[1:-1] - phi * x[:-2]) / sigmasq - phi * (x[2:] - phi * x[1:-1]) / sigmasq
    r = np.append(do, w)
    r = np.append(r, dn)
    glp = s - r
    return glp

def volatility_metric(sigmasq: float, phi: float, phisq: float, T: int):
    r = -phi / sigmasq * np.ones((T-1))
    fl = 1. / sigmasq
    m = (1.0 + phisq) / sigmasq * np.ones(T-2)
    G = np.zeros([2, T])
    G[1] = np.hstack((fl, m, fl)) + 0.5
    G[0, 1:] = r

    # One can construct the full Riemannian metric as follows:
    # >>> a = np.diag(r, 1)
    # >>> b = np.diag(r, -1)
    # >>> c = np.diag(np.hstack([fl, m, fl]))
    # >>> iC = a + b + c
    # >>> G = iC
    # >>> G[np.diag_indices_from(G)] += 0.5

    # One can check the calculation of the inverse of the AR(1) covariance
    # using:
    # >>> n = np.arange(T) + 1
    # >>> C = np.power(phi, np.abs(n - n[..., np.newaxis])) * sigmasq / (1 - phisq)
    # >>> assert np.allclose(iC, np.linalg.inv(C))
    return G

def volatility_posterior_factory(sigma: float, phi: float, beta: float, y: np.ndarray) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of the stochastic volatility model given parameters `sigma`, `phi`, and
    `beta`. The factory also constructs functions for the gradient of the
    log-posterior and the Fisher information metric.

    Args:
        sigma: Parameter of the stochastic volatility model.
        phi: Parameter of the stochastic volatility model.
        beta: Parameter of the stochastic volatility model.
        y: Observations from the stochastic volatility model.

    Returns:
        euclidean_auxiliaries: Function to compute the log-posterior and the
            gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.

    """
    sigmasq = np.square(sigma)
    phisq = np.square(phi)
    T = y.size
    S = T - 1
    ysq = np.square(y)

    def euclidean_auxiliaries(x: np.ndarray) -> Tuple[np.ndarray]:
        """Computes the log-posterior and the gradient of the log-posterior of the
        stochastic volatility model over the volatilities given fixed values of
        the model hyperparameters.

        Args:
            x: The stochastic volatilities.

        Returns:
            lp: The log-posterior of the stochastic volatility model.
            glp: The gradient of the log-posterior of the stochastic volatility
                model.

        """
        omega = beta*np.exp(0.5*x)
        omegasq = np.square(omega)
        lp = volatility_log_posterior(x, omega, omegasq, sigma, sigmasq, phi, phisq, beta, T, S, ysq)
        glp = volatility_grad_log_posterior(x, omega, omegasq, sigma, sigmasq, phi, phisq, beta, T, S, ysq)
        return lp, glp

    def metric() -> np.ndarray:
        """The Fisher information metric for the stochastic volatility model given
        values of `phi`, `sigma`, and `beta`.

        Returns:
            G: The Fisher information metric.

        """
        G = volatility_metric(sigmasq, phi, phisq, T)
        return G

    return euclidean_auxiliaries, metric

def base_quantities(qt: np.ndarray) -> Tuple[float, ...]:
    gamma, alpha, beta = qt
    sigma = np.exp(gamma)
    sigmasq = np.square(sigma)
    phi = np.tanh(alpha)
    phisq = np.square(phi)
    return gamma, alpha, beta, sigma, sigmasq, phi, phisq

def hyperparameter_log_posterior(sigma: float,
                                 phi: float,
                                 beta: float,
                                 T: int,
                                 S: int,
                                 x: np.ndarray,
                                 ysq: np.ndarray) -> float:
    omega = beta*np.exp(0.5*x)
    omegasq = np.square(omega)
    ly = -0.5*np.sum(ysq / omegasq) - np.sum(np.log(omega)) - 0.5*T*np.log(2*np.pi)
    sigma_xo = sigma / np.sqrt(1.0 - np.square(phi))
    lxo = -0.5*np.square(x[0] / sigma_xo) - np.log(sigma_xo) - 0.5*np.log(2*np.pi)
    lx = -0.5*np.sum(np.square((x[1:] - phi*x[:-1]) / sigma)) - S*np.log(sigma) - 0.5*S*np.log(2*np.pi)
    lp = ly + lx + lxo + prior.log_prior(sigma, phi, beta)
    return lp

def hyperparameter_grad_log_posterior(gamma: float,
                                      alpha: float,
                                      beta: float,
                                      sigmasq: float,
                                      phi: float,
                                      phisq: float,
                                      T: int,
                                      x: np.ndarray,
                                      ysq: np.ndarray) -> np.ndarray:
    dpgamma, dpalpha, dpbeta = prior.grad_log_prior(gamma, alpha, beta)
    dbeta = (-T / beta
             + np.sum(ysq / np.exp(x)) / np.power(beta, 3.0)
             + dpbeta)
    dgamma = (
        -T + np.square(x[0])*(1.0 - phisq) / sigmasq
        + np.sum(np.square(x[1:] - phi*x[:-1])) / sigmasq
        + dpgamma)
    dalpha = (
        -phi + phi*np.square(x[0])*(1.0 - phisq) / sigmasq
        + np.sum(x[:-1] * (x[1:] - phi*x[:-1])) * (1.0 - phisq) / sigmasq
        + dpalpha)
    return np.array((dgamma, dalpha, dbeta))

def hyperparameter_metric(gamma: float,
                          alpha: float,
                          beta: float,
                          sigmasq: float,
                          phi: float,
                          phisq: float,
                          T: int) -> np.ndarray:
    # Note that this ordering of the variables differs from that presented in
    # the Riemannian manifold HMC paper. These quantities are derived in the
    # RMHMC paper in equations (27) and (28).
    G = np.array([
        #  gamma                                alpha                       beta
        [  2.0*T,                             2.0*phi,                       0.0], # gamma
        [2.0*phi, 2.0*phisq + (T - 1.0)*(1.0 - phisq),                       0.0], # alpha
        [    0.0,                                 0.0, 2.0 * T / np.square(beta)]  # beta
    ])
    # Add in the negative Hessian of the log-prior.
    H = prior.hess_log_prior(gamma, alpha, beta)
    G = G - H
    return G

def hyperparameter_grad_metric(gamma: float,
                               alpha: float,
                               beta: float,
                               sigmasq: float,
                               phi: float,
                               phisq: float,
                               T: int) -> np.ndarray:
    dGbeta = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -4.0 * T / np.power(beta, 3.0)]
    ])
    dGgamma = np.zeros((3, 3))
    a = 2.0*(1.0 - phisq)
    b = 2.0*phi*(3.0 - T)*(1.0 - phisq)
    dGalpha = np.array([
        [0.0,   a, 0.0],
        [  a,   b, 0.0],
        [0.0, 0.0, 0.0]
    ])
    dG = np.stack((dGgamma, dGalpha, dGbeta), axis=0)
    dG = np.transpose(dG, (2, 1, 0))
    dH = prior.grad_hess_log_prior(gamma, alpha, beta)
    return dG - dH


def hyperparameter_posterior_factory(x: np.ndarray, y: np.ndarray) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of the stochastic volatility model given parameters `x`. The factory also
    constructs functions for the gradient of the log-posterior and the Fisher
    information metric.

    Args:
        x: The stochastic volatilities.
        y: Observations from the stochastic volatility model.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.

    """
    T = x.size
    S = T - 1
    ysq = np.square(y)

    def log_posterior(qt: np.ndarray) -> float:
        """The log-posterior of the stochastic volatility model given the stochastic
        volatilities. The inference is over the model parameters `sigma`, `phi`,
        and `beta`.

        Args:
            qt: An array containing the transformed hyperparameters of the
                stochastic volatility model.

        Returns:
            lp: The log-posterior of the stochastic volatility model.

        """
        gamma, alpha, beta, sigma, sigmasq, phi, phisq = base_quantities(qt)
        lp = hyperparameter_log_posterior(sigma, phi, beta, T, S, x, ysq)
        return lp

    def metric(qt: np.ndarray) -> np.ndarray:
        """The Fisher information metric (plus the negative Hessian of the log-prior)
        of the stochastic volatility model.

        Args:
            qt: An array containing the transformed hyperparameters of the
                stochastic volatility model.

        Returns:
            G: The Riemannian metric.

        """
        gamma, alpha, beta, sigma, sigmasq, phi, phisq = base_quantities(qt)
        G = hyperparameter_metric(gamma, alpha, beta, sigmasq, phi, phisq, T)
        return G

    def euclidean_auxiliaries(qt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma, alpha, beta, sigma, sigmasq, phi, phisq = base_quantities(qt)
        lp = hyperparameter_log_posterior(sigma, phi, beta, T, S, x, ysq)
        glp = hyperparameter_grad_log_posterior(gamma, alpha, beta, sigmasq, phi, phisq, T, x, ysq)
        return lp, glp

    def riemannian_auxiliaries(qt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gamma, alpha, beta, sigma, sigmasq, phi, phisq = base_quantities(qt)
        lp = hyperparameter_log_posterior(sigma, phi, beta, T, S, x, ysq)
        glp = hyperparameter_grad_log_posterior(gamma, alpha, beta, sigmasq, phi, phisq, T, x, ysq)
        G = hyperparameter_metric(gamma, alpha, beta, sigmasq, phi, phisq, T)
        dG = hyperparameter_grad_metric(gamma, alpha, beta, sigmasq, phi, phisq, T)
        return lp, glp, G, dG

    def log_posterior_and_metric(qt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma, alpha, beta, sigma, sigmasq, phi, phisq = base_quantities(qt)
        lp = hyperparameter_log_posterior(sigma, phi, beta, T, S, x, ysq)
        G = hyperparameter_metric(gamma, alpha, beta, sigmasq, phi, phisq, T)
        return lp, G

    return log_posterior, metric, log_posterior_and_metric, euclidean_auxiliaries, riemannian_auxiliaries
