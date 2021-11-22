import unittest

import numpy as np

from hmc.applications.cox_poisson import forward_transform, inverse_transform, generate_data, gaussian_posterior_factory, hyperparameter_posterior_factory
from hmc.applications.cox_poisson.prior import log_prior, grad_log_prior, hess_log_prior, grad_hess_log_prior


class TestCoxPoisson(unittest.TestCase):
    def test_prior(self):
        def transformed_log_prior(qt):
            return log_prior(*inverse_transform(qt)[0])

        transformed_grad_log_prior = lambda qt: grad_log_prior(*qt)
        transformed_hess_log_prior = lambda qt: hess_log_prior(*qt)
        transformed_grad_hess_log_prior = lambda qt: grad_hess_log_prior(*qt)

        q = np.random.uniform(size=(2, ))
        qt, _ = forward_transform(q)

        delta = 1e-5

        u = np.random.normal(size=qt.shape)
        fd = (transformed_log_prior(qt + 0.5*delta*u) - transformed_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_grad_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (transformed_grad_log_prior(qt + 0.5*delta*u) - transformed_grad_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_hess_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (transformed_hess_log_prior(qt + 0.5*delta*u) - transformed_hess_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_grad_hess_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

    def test_gaussian_posterior(self):
        sigmasq, beta = np.random.uniform(size=(2, ))
        mu = np.log(126.0) - sigmasq / 2.0
        dist, x, y = generate_data(10, mu, beta, sigmasq)

        euclidean_auxiliaries, metric = gaussian_posterior_factory(dist, mu, sigmasq, beta, y)
        log_posterior = lambda x: euclidean_auxiliaries(x)[0]
        grad_log_posterior = lambda x: euclidean_auxiliaries(x)[1]
        delta = 1e-6

        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

    def test_hyperparameter_posterior(self):
        sigmasq, beta = np.random.uniform(size=(2, ))
        mu = np.log(126.0) - sigmasq / 2.0
        dist, x, y = generate_data(16, mu, beta, sigmasq)

        log_posterior, metric, _, euclidean_auxiliaries, riemannian_auxiliaries = hyperparameter_posterior_factory(dist, mu, x, y)

        grad_log_posterior = lambda qt: euclidean_auxiliaries(qt)[1]
        grad_metric = lambda qt: riemannian_auxiliaries(qt)[3]

        q = np.array([sigmasq, beta])
        qt, _ = forward_transform(q)

        delta = 1e-4
        u = np.random.normal(size=(2, ))
        fd = (log_posterior(qt + 0.5*delta*u) - log_posterior(qt - 0.5*delta*u)) / delta
        dd = grad_log_posterior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (metric(qt + 0.5*delta*u) - metric(qt - 0.5*delta*u)) / delta
        dd = grad_metric(qt)@u
        self.assertTrue(np.allclose(fd, dd))
