from typing import Callable, Tuple

import numpy as np

from .leapfrog_state import LeapfrogState
from hmc.integrators.fields import riemannian, softabs
from hmc.linalg import solve_psd, eigen_decomposition_to_matrix


class SoftAbsLeapfrogState(LeapfrogState):
    """The SoftAbs metric is an general metric that is compatible with Riemannian
    Hamiltonian Monte Carlo. It is computed by taking the Hessian and applying
    a smooth, positive function to the eigenvalues of the Hessian that
    approximates the absolute value function.

    """
    def __init__(self,
                 position: np.ndarray,
                 momentum: np.ndarray,
                 alpha: float):
        super().__init__(position, momentum)
        self.alpha = alpha
        self._hessian: np.ndarray
        self._hessian_eigenvals: np.ndarray
        self._hessian_eigenvecs: np.ndarray
        self._jac_hessian: np.ndarray
        self._softabs_eigenvals: np.ndarray
        self._softabs_inv_eigenvals: np.ndarray

    @property
    def requires_update(self) -> bool:
        o = self.log_posterior is None or \
            self.grad_log_posterior is None or \
            self.metric is None or \
            self.inv_metric is None or \
            self.hessian is None or \
            self.hessian_eigenvals is None or \
            self.hessian_eigenvecs is None or \
            self.jac_hessian is None or \
            self.softabs_eigenvals is None or \
            self.softabs_inv_eigenvals is None
        return o

    @property
    def hessian(self):
        return self._hessian

    @hessian.setter
    def hessian(self, value):
        self._hessian = value

    @hessian.deleter
    def hessian(self):
        del self._hessian

    @property
    def hessian_eigenvals(self):
        return self._hessian_eigenvals

    @hessian_eigenvals.setter
    def hessian_eigenvals(self, value):
        self._hessian_eigenvals = value

    @hessian_eigenvals.deleter
    def hessian_eigenvals(self):
        del self._hessian_eigenvals

    @property
    def hessian_eigenvecs(self):
        return self._hessian_eigenvecs

    @hessian_eigenvecs.setter
    def hessian_eigenvecs(self, value):
        self._hessian_eigenvecs = value

    @hessian_eigenvecs.deleter
    def hessian_eigenvecs(self):
        del self._hessian_eigenvecs

    @property
    def jac_hessian(self):
        return self._jac_hessian

    @jac_hessian.setter
    def jac_hessian(self, value):
        self._jac_hessian = value

    @jac_hessian.deleter
    def jac_hessian(self):
        del self._jac_hessian

    @property
    def softabs_eigenvals(self):
        return self._softabs_eigenvals

    @softabs_eigenvals.setter
    def softabs_eigenvals(self, value):
        self._softabs_eigenvals = value

    @softabs_eigenvals.deleter
    def softabs_eigenvals(self):
        del self._softabs_eigenvals

    @property
    def softabs_inv_eigenvals(self):
        return self._softabs_inv_eigenvals

    @softabs_inv_eigenvals.setter
    def softabs_inv_eigenvals(self, value):
        self._softabs_inv_eigenvals = value

    @softabs_inv_eigenvals.deleter
    def softabs_inv_eigenvals(self):
        del self._softabs_inv_eigenvals

    def update(self, auxiliaries: Callable):
        lp, glp, H, dH = auxiliaries(self.position)
        l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, self.alpha)
        self.log_posterior = lp
        self.grad_log_posterior = glp
        self.hessian = H
        self.hessian_eigenvals = l
        self.hessian_eigenvecs = U
        self.jac_hessian = dH
        self.softabs_eigenvals = lt
        self.softabs_inv_eigenvals = inv_lt
        self.metric = metric
        self.inv_metric = inv_metric
        self.velocity = riemannian.velocity(inv_metric, self.momentum)
        self.force = softabs.force(
            self.momentum, glp, dH, l, lt, inv_lt, U, self.alpha)

    def clear(self):
        super().clear()
        del self.hessian
        del self.hessian_eigenvals
        del self.hessian_eigenvecs
        del self.jac_hessian
        del self.softabs_eigenvals
        del self.softabs_inv_eigenvals
