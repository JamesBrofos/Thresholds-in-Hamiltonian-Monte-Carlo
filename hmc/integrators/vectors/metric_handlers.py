from typing import Callable

import numpy as np

from hmc.linalg import solve_psd, sqrtm
from hmc.integrators.fields import softabs


def euclidean_metric_handler(metric: np.ndarray) -> Callable:
    """Convenience function for computing an constant Euclidean metric, its matrix
    square-root, and log-determinant.

    """
    inv_metric, chol_metric = solve_psd(metric, return_chol=True)
    logdet_metric = 2.0*np.sum(np.log(np.diag(chol_metric)))
    def metric_handler(state):
        state.metric = metric
        state.inv_metric = inv_metric
        state.sqrtm_metric = chol_metric
        state.logdet_metric = logdet_metric
    return metric_handler

def riemannian_metric_handler(metric: Callable) -> Callable:
    """Convenience function for computing the metric, its inverse, its square root,
    and its log-determinant for a Riemannian metric system.

    """
    def metric_handler(state):
        G = metric(state.position)
        inv_metric, chol_metric = solve_psd(G, return_chol=True)
        logdet_metric = 2.0*np.sum(np.log(np.diag(chol_metric)))
        state.metric = G
        state.inv_metric = inv_metric
        state.sqrtm_metric = chol_metric
        state.logdet_metric = logdet_metric
    return metric_handler

def softabs_metric_handler(hessian: Callable, alpha: float) -> Callable:
    """Convenience function for computing the metric, its inverse, its square root,
    and its log-determinant for a SoftAbs metric system.

    """
    def metric_handler(state):
        H = hessian(state.position)
        _, U, lt, _, metric, inv_metric = softabs.decomposition(H, alpha)
        sqrtm_metric = sqrtm(lt, U)
        logdet_metric = np.sum(np.log(lt))
        state.metric = metric
        state.inv_metric = inv_metric
        state.sqrtm_metric = sqrtm_metric
        state.logdet_metric = logdet_metric
    return metric_handler

