import unittest

import numpy as np

from hmc.applications.t import posterior_factory, sample

class TestStudentT(unittest.TestCase):
    def test_posterior(self):
        n = int(np.ceil(20*np.random.uniform()))
        dof = int(np.ceil(50*np.random.uniform()))
        L = np.random.normal(size=(n, n))
        Sigma = L@L.T
        log_posterior, metric, log_posterior_and_metric, euclidean_auxiliaries, riemannian_auxiliaries = posterior_factory(Sigma, dof)

        grad_log_posterior = lambda x: euclidean_auxiliaries(x)[1]
        grad_metric = lambda x: riemannian_auxiliaries(x)[3]
        delta = 1e-5

        x = sample(L, dof)
        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

        # fd = (grad_log_posterior(x + 0.5*delta*u) - grad_log_posterior(x - 0.5*delta*u)) / delta
        # dd = -metric(x)@u
        # self.assertTrue(np.allclose(fd, dd))

        fd = (metric(x + 0.5*delta*u) - metric(x - 0.5*delta*u)) / delta
        dd = grad_metric(x)@u
        self.assertTrue(np.allclose(fd, dd))

        G = metric(x)
        self.assertTrue(np.allclose(G, G.T))
        w, v = np.linalg.eigh(G)
        self.assertTrue(np.all(w > 0.0))
