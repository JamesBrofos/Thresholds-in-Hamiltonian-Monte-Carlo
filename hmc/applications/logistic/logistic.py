from typing import Callable, Tuple

import numpy as np


# Sigmoid function and its derivatives.
def sigmoid(x):
    """This implementation of the sigmoid function is from [1].

    [1] https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python

    """
    idx = x >= 0.0
    z = np.zeros(len(x))
    z[idx] = 1.0 / (1.0 + np.exp(-x[idx]))
    p = np.exp(x[~idx])
    z[~idx] = p / (1.0 + p)
    return z

def sigmoid_p(z):
    p = sigmoid(z)
    return p*(1.0 - p)

def sigmoid_pp(z):
    p = sigmoid(z)
    pp = sigmoid_p(z)
    return pp - 2*p*pp


def log_posterior_helper(lin: np.ndarray, beta: np.ndarray, y: np.ndarray, inv_alpha: float):
    ll = -np.sum(np.maximum(lin, 0.0) - lin*y + np.log(1.0 + np.exp(-np.abs(lin))))
    lbeta = -0.5 * inv_alpha * np.sum(np.square(beta))
    lp = ll + lbeta
    return lp

def grad_log_posterior_helper(lin: np.ndarray, beta: np.ndarray, x: np.ndarray, y: np.ndarray, inv_alpha: float) -> np.ndarray:
    yp = sigmoid(lin)
    glp = (y - yp)@x - inv_alpha * beta
    return glp

def metric_helper(lin: np.ndarray, x: np.ndarray, inv_alpha: float) -> np.ndarray:
    L = sigmoid_p(lin)
    G = (x.T*L)@x + inv_alpha * np.eye(x.shape[-1])
    return G

def grad_metric_helper(lin: np.ndarray, x: np.ndarray) -> np.ndarray:
    o = sigmoid_pp(lin)
    Q = o[..., np.newaxis] * x
    dG = x.T@(Q[..., np.newaxis] * x[:, np.newaxis]).swapaxes(0, 1)
    return dG

def sample_posterior_precision(beta: np.ndarray, k: float, theta: float) -> float:
    """Samples the posterior distribution of the precision parameter, which can be
    shown to be a Gamma distribution with a prescribed shape and scale.

    Args:
        beta: The current linear coefficients.
        k: The shape of the precision Gamma prior.
        theta: The scale of the precision Gamma prior.

    Returns:
        inv_alpha: The precision parameter.

    """
    d = len(beta)
    shape = k + 0.5*d
    scale = np.reciprocal(0.5*np.sum(np.square(beta)) + np.reciprocal(theta))
    inv_alpha = np.random.gamma(shape, scale)
    return inv_alpha

def posterior_factory(x: np.ndarray, y: np.ndarray, inv_alpha: float) -> Tuple[Callable]:
    """Factory function that yields further functions to compute the log-posterior
    of a Bayesian logistic regression model, the gradient of the log-posterior,
    the Fisher information metric, and the gradient of the Fisher information
    metric.

    Args:
        x: Covariates of the logistic regression.
        y: Binary targets of the logistic regression.
        inv_alpha: The precision of the normal prior over the linear coefficients.

    Returns:
        log_posterior: Function to compute the log-posterior.
        metric: Function to compute the Fisher information metric.
        euclidean_auxiliaries: Function to compute the log-posterior and its
            gradient.
        riemannian_auxiliaries: Function to compute the log-posterior, the
            gradient of the log-posterior, the Fisher information metric, and the
            derivatives of the Fisher information metric.

    """
    def log_posterior(beta: np.ndarray) -> float:
        """Log-posterior of a Bayesian logistic regression with a Bernoulli likelihood
        and a normal prior over the linear coefficients.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            lp: Log-posterior of the Bayesian logistic regression.

        """
        return log_posterior_helper(x@beta, beta, y, inv_alpha)

    def metric(beta: np.ndarray) -> np.ndarray:
        """Fisher information metric for Bayesian logistic regression model.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            G: The Fisher information metric of the Bayesian logistic regression
                model.

        """
        return metric_helper(x@beta, x, inv_alpha)

    def euclidean_auxiliaries(beta: np.ndarray) -> Tuple[np.ndarray]:
        """Function to compute the log-posterior and the gradient of the
        log-posterior.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            lp: The log-posterior of the logistic regression model.
            glp: The gradient of the log-posterior of the logistic regression
                model.

        """
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, inv_alpha)
        glp = grad_log_posterior_helper(lin, beta, x, y, inv_alpha)
        return lp, glp

    def riemannian_auxiliaries(beta: np.ndarray) -> Tuple[np.ndarray]:
        """Function to compute the log-posterior, the gradient of the log-posterior,
        the Fisher information metric and the derivatives of the Fisher
        information metric.

        Args:
            beta: Linear coefficients of the logistic regression.

        Returns:
            lp: The log-posterior of the logistic regression model.
            glp: The gradient of the log-posterior of the logistic regression
                model.
            G: The Fisher information metric of the logistic regression model.
            dG: The gradient of the Fisher information metric with respect to the
                linear coefficients.

        """
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, inv_alpha)
        glp = grad_log_posterior_helper(lin, beta, x, y, inv_alpha)
        G = metric_helper(lin, x, inv_alpha)
        dG = grad_metric_helper(lin, x)
        return lp, glp, G, dG

    def log_posterior_and_metric(beta: np.ndarray) -> Tuple[np.ndarray]:
        lin = x@beta
        lp = log_posterior_helper(lin, beta, y, inv_alpha)
        G = metric_helper(lin, x, inv_alpha)
        return lp, G

    return log_posterior, metric, log_posterior_and_metric, euclidean_auxiliaries, riemannian_auxiliaries
