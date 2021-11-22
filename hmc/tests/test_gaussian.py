import unittest

import numpy as np

from hmc.applications.gaussian import posterior_factory

class TestGaussian(unittest.TestCase):
    def test_posterior(self):
        n = int(np.ceil(100*np.random.uniform()))

        L = np.random.normal(size=(n, n))
        Sigma = L@L.T
        mu = np.random.normal(size=n)
        euclidean_auxiliaries, metric = posterior_factory(mu, Sigma)
        x = np.random.multivariate_normal(mu, Sigma)

        log_posterior = lambda x: euclidean_auxiliaries(x)[0]
        grad_log_posterior = lambda x: euclidean_auxiliaries(x)[1]
        delta = 1e-4

        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (grad_log_posterior(x + 0.5*delta*u) - grad_log_posterior(x - 0.5*delta*u)) / delta
        dd = -metric(x)@u
        self.assertTrue(np.allclose(fd, dd))
