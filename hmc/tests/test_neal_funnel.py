import unittest

import numpy as np

from hmc.applications.neal_funnel import posterior_factory, sample
from hmc.integrators.states import SoftAbsLeapfrogState
from hmc.integrators.fields import riemannian
from hmc.integrators.fields import softabs
from hmc.hamiltonian import hamiltonian


class TestNealFunnel(unittest.TestCase):
    def test_auxiliaries(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        log_density, hess_log_density, euclidean_auxiliaries, riemannian_auxiliaries = posterior_factory()
        grad_log_density = lambda q: euclidean_auxiliaries(q)[1]

        x, v = sample(num_dims)
        q = np.hstack((x, v))
        delta = 1e-5
        u = np.random.normal(size=q.shape)
        fd = (log_density(q + 0.5*delta*u) - log_density(q - 0.5*delta*u)) / delta
        _, glp, H, dH = riemannian_auxiliaries(q)
        self.assertTrue(np.allclose(fd, glp@u))
        fd = (grad_log_density(q + 0.5*delta*u) - grad_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, H@u))
        fd = (hess_log_density(q + 0.5*delta*u) - hess_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, dH@u))

    def test_hamiltonian(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        x, v = sample(num_dims)
        q = np.hstack((x, v))
        log_density, hess_log_density, euclidean_auxiliaries, riemannian_auxiliaries = posterior_factory()
        alpha = 1e1
        state = SoftAbsLeapfrogState(q, np.zeros_like(q), alpha)
        state.update(riemannian_auxiliaries)
        L = np.linalg.cholesky(state.metric)
        p = L@np.random.normal(size=q.shape)
        self.assertTrue(np.allclose(L@L.T, state.metric))

        def _hamiltonian(q, p):
            ld = log_density(q)
            _, _, H, _ = riemannian_auxiliaries(q)
            l, U, lt, inv_lt, metric, inv_metric= softabs.decomposition(H, alpha)
            logdet = np.sum(np.log(lt))
            ham = hamiltonian(p, ld, logdet, inv_metric)
            return ham

        delta = 1e-5
        u = np.random.normal(size=q.shape)
        fd = (_hamiltonian(q, p+0.5*delta*u) - _hamiltonian(q, p-0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, riemannian.velocity(state.inv_metric, p)@u))
        fd = (_hamiltonian(q+0.5*delta*u, p) - _hamiltonian(q-0.5*delta*u, p)) / delta
        dHdq = -softabs.force(p,
                              state.grad_log_posterior,
                              state.jac_hessian,
                              state.hessian_eigenvals,
                              state.softabs_eigenvals,
                              state.softabs_inv_eigenvals,
                              state.hessian_eigenvecs,
                              alpha)@u
        self.assertTrue(np.allclose(fd, dHdq))
