from typing import Callable, Tuple

import numpy as np


def sample(num_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from Neal's funnel distribution.

    Args:
        num_dims: Number of dimensions, besides the global variance, in Neal's
            funnel distribution.

    Returns:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.

    """
    v = np.random.normal(0.0, 3.0)
    s = np.exp(-0.5*v)
    x = np.random.normal(0.0, s, size=(num_dims, ))
    return x, v

def _log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> float:
    """Log-density of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        ld: The log-density of Neal's funnel.

    """
    ldx = -0.5*ssx_div_ssq - 0.5*num_dims*np.log(2*np.pi) - num_dims*np.log(s)
    ldv = -0.5*np.square(v) / 9.0 - 0.5*np.log(2*np.pi) - np.log(3.0)
    ld = ldv + ldx
    return ld

def _grad_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Gradient of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        out: The gradient of the log-density of Neal's funnel.

    """
    glp = np.hstack([
        -x_div_ssq,
        -v/9.0 - 0.5 * ssx_div_ssq + 0.5 * num_dims])
    return glp

def _hess_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        H: The Hessian of the log-density of Neal's funnel.

    """
    dvdv = -1.0/9.0 - 0.5 * ssx_div_ssq
    dvdx = -x_div_ssq
    dxdx = -np.eye(num_dims) / ssq
    H = np.vstack((
        np.hstack((dxdx, dvdx[..., np.newaxis])),
        np.hstack((dvdx, dvdv))
    ))
    return H

def _grad_hess_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Gradient of the Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        dH: The higher-order derivatives of the log-density of Neal's funnel.

    """
    dvdvdv = -0.5 * ssx_div_ssq
    dxdxdv = -np.eye(num_dims) / ssq
    dvdvdx = -x_div_ssq

    Z = np.zeros((num_dims, num_dims, num_dims))
    da = np.concatenate((Z, dxdxdv[..., np.newaxis]), axis=-1)
    mm = np.hstack((dxdxdv, dvdvdx[..., np.newaxis]))
    rr = np.vstack((mm, np.hstack((dvdvdx, dvdvdv))))
    db = np.concatenate((da, mm[:, np.newaxis]), axis=1)
    dH = np.concatenate((db, rr[np.newaxis]))
    return dH

separator = lambda q: (q[:-1], q[-1])


def posterior_factory() -> Tuple[Callable]:
    """Posterior factory function for Neal's funnel distribution. This is a density
    that exhibits extreme variation in the dimensions and may therefore present
    a challenge for leapfrog integrators. Therefore, the posterior is also
    equipped with the softabs metric which adapts the generalized leapfrog
    integrator to the local geometry. The softabs metric is a transformation of
    the Hessian to make it positive definite.

    It is a curious attribute of this posterior that for a larger size of the
    posterior, larger step-sizes are better behaved that for a smaller size of
    the posterior.

    Returns:
        euclidean_auxiliaries: Function to compute the log-density and its
            gradient.
        riemannian_auxiliaries: Function to compute the log-density, the gradient
            of the log-density, the Hessian of the log-density and the gradient
            of the Hessian of the log-density.

    """
    def _base_quantities(q):
        x, v = separator(q)
        num_dims = x.size
        s = np.exp(-0.5*v)
        ssq = np.square(s)
        x_div_ssq = x / ssq
        ssx_div_ssq = np.square(x).sum() / ssq
        return x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq

    def log_density(q):
        x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq = _base_quantities(q)
        ld = _log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return ld

    def hessian(q):
        x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq = _base_quantities(q)
        H = _hess_log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return H

    def euclidean_auxiliaries(q):
        x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq = _base_quantities(q)
        lp = _log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        glp = _grad_log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return lp, glp

    def riemannian_auxiliaries(q):
        x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq = _base_quantities(q)
        lp = _log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        glp = _grad_log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        H = _hess_log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        dH = _grad_hess_log_density(x, v, num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return lp, glp, H, dH

    return log_density, hessian, euclidean_auxiliaries, riemannian_auxiliaries
