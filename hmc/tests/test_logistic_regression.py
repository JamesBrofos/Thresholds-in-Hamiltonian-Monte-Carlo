import unittest

import numpy as np

from hmc.applications.logistic import posterior_factory, hierarchical_posterior_factory, sigmoid


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        # Generate logistic regression data.
        num_obs, num_dims = 100, 5
        x = np.random.normal(size=(num_obs, num_dims))
        b = np.ones((x.shape[-1], ))
        p = sigmoid(x@b)
        y = np.random.binomial(1, p)
        alpha = 0.5

        # Check the gradients of the posterior using finite differences.
        (
            log_posterior, metric, _, euclidean_auxiliaries, riemannian_auxiliaries
        ) = posterior_factory(x, y, alpha)
        grad_log_posterior = lambda q: euclidean_auxiliaries(q)[1]
        grad_metric = lambda q: riemannian_auxiliaries(q)[3]
        delta = 1e-6
        u = np.random.normal(size=b.shape)
        fd = (log_posterior(b + 0.5*delta*u) - log_posterior(b - 0.5*delta*u)) / delta
        dd = grad_log_posterior(b)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(b + 0.5*delta*u) - metric(b - 0.5*delta*u)) / delta
        dG = grad_metric(b)
        self.assertTrue(np.allclose(fd, dG@u))

    def test_hierarchical(self):
        num_obs, num_dims = 100, 2
        x = np.random.normal(size=(num_obs, num_dims))
        b = np.ones((x.shape[-1], ))
        p = sigmoid(x@b)
        y = np.random.binomial(1, p)
        lam = 2.0
        ialpha = np.random.exponential(1.0 / lam)

        log_posterior, hessian, euclidean_auxiliaries, riemannian_auxiliaries = hierarchical_posterior_factory(x, y, lam)
        grad_log_posterior = lambda q: euclidean_auxiliaries(q)[1]
        hess_log_posterior = hessian
        grad_hess_log_posterior = lambda q: riemannian_auxiliaries(q)[3]

        q = np.hstack((b, ialpha))
        riemannian_auxiliaries(q)

        delta = 1e-6
        u = np.random.normal(size=q.shape)
        fd = (log_posterior(q + 0.5*delta*u) - log_posterior(q - 0.5*delta*u)) / delta
        dd = grad_log_posterior(q)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (grad_log_posterior(q + 0.5*delta*u) - grad_log_posterior(q - 0.5*delta*u)) / delta
        H = hess_log_posterior(q)
        dd = H@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (hess_log_posterior(q + 0.5*delta*u) - hess_log_posterior(q - 0.5*delta*u)) / delta
        dd = grad_hess_log_posterior(q)@u
        self.assertTrue(np.allclose(fd, dd))
