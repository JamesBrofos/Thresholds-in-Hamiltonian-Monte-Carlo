from typing import Callable

import numpy as np

from hmc.linalg import solve_psd


def newton_raphson(q: np.ndarray, riemannian_auxiliaries: Callable, tol: float=1e-10) -> np.ndarray:
    """Implements the Newton-Raphson algorithm to find the maximum a posteriori of
    the posterior.

    Args:
        q: Initial guess for the location of the maximum of the posterior.
        riemannian_auxiliaries: Function to compute the log-posterior, the
            gradient of the log-posterior, the Fisher information metric, and the
            derivatives of the Fisher information metric.
        tol: The convergence tolerance for Newton iterations.

    Returns:
        q: The maximizer of the posterior density.

    """
    delta = np.inf
    while delta > tol:
        _, g, G, _ = riemannian_auxiliaries(q)
        q += solve_psd(G, g)
        delta = np.abs(g).max()
    return q
