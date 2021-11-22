import unittest

import numpy as np

from hmc.applications.stochastic_volatility import generate_data, \
    volatility_posterior_factory, hyperparameter_posterior_factory, \
    forward_transform, inverse_transform
from hmc.applications.stochastic_volatility.prior import log_prior, grad_log_prior, hess_log_prior, grad_hess_log_prior
from hmc.linalg import solve_tridiagonal, cholesky_tridiagonal


class TestStochasticVolatility(unittest.TestCase):
    def test_volatility_posterior(self):
        sigma, phi, beta = np.random.uniform(size=(3, ))
        T = int(np.ceil(1000*np.random.uniform()))
        x, y = generate_data(T, sigma, phi, beta)

        euclidean_auxiliaries, metric = volatility_posterior_factory(sigma, phi, beta, y)
        log_posterior = lambda x: euclidean_auxiliaries(x)[0]
        grad_log_posterior = lambda x: euclidean_auxiliaries(x)[1]
        delta = 1e-6

        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

        G = metric()
        _G = np.diag(G[1]) + np.diag(G[0, 1:], -1) + np.diag(G[0, 1:], 1)
        rhs = np.random.normal(size=x.shape)
        sol = solve_tridiagonal(G, rhs)
        self.assertTrue(np.allclose(_G@sol, rhs))

        L = cholesky_tridiagonal(G)
        self.assertTrue(np.allclose((L@L.T).toarray(), _G))

        iG = solve_tridiagonal(G)
        self.assertTrue(np.allclose(_G@iG, np.eye(T)))

    def test_prior(self):
        def transformed_log_prior(qt):
            return log_prior(*inverse_transform(qt)[0])

        transformed_grad_log_prior = lambda qt: grad_log_prior(*qt)
        transformed_hess_log_prior = lambda qt: hess_log_prior(*qt)
        transformed_grad_hess_log_prior = lambda qt: grad_hess_log_prior(*qt)

        q = np.random.uniform(size=(3, ))
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

    def test_hyperparameter_posterior(self):
        sigma, phi, beta = np.random.uniform(size=(3, ))
        T = int(np.ceil(1000*np.random.uniform()))
        x, y = generate_data(T, sigma, phi, beta)
        q = np.array([sigma, phi, beta])
        qt = forward_transform(q)[0]

        log_posterior, metric, _, euclidean_auxiliaries, riemannian_auxiliaries = hyperparameter_posterior_factory(x, y)
        grad_log_posterior = lambda x: euclidean_auxiliaries(x)[1]
        grad_metric = lambda x: riemannian_auxiliaries(x)[3]
        delta = 1e-6

        u = np.random.normal(size=qt.shape)
        fd = (log_posterior(qt + 0.5*delta*u) - log_posterior(qt - 0.5*delta*u)) / delta
        dd = grad_log_posterior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (metric(qt + 0.5*delta*u) - metric(qt - 0.5*delta*u)) / delta
        dd = grad_metric(qt)@u
        self.assertTrue(np.allclose(fd, dd))
