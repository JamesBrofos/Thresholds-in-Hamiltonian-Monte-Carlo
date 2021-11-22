import unittest

import numpy as np

import hmc
from hmc.applications import banana as distr
from hmc.integrators.vectors import velocity_and_force_from_riemannian_auxiliaries
from hmc.integrators.states import RiemannianLeapfrogState, VectorFieldLeapfrogState
from hmc.integrators.stateful.generalized_leapfrog import generalized_leapfrog
from hmc.integrators.vectors import vector_field_leapfrog, riemannian_metric_handler


class TestGeneralizedLeapfrog(unittest.TestCase):
    def test_integrator(self):
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = distr.generate_data(t, sigma_y, sigma_theta, 100)
        log_posterior, metric, _, euclidean_auxiliaries, riemannnian_auxiliaries = \
            distr.posterior_factory(y, sigma_y, sigma_theta)
        velocity_vector, force_vector = velocity_and_force_from_riemannian_auxiliaries(
            metric, riemannnian_auxiliaries
        )

        q = np.array([t, np.sqrt(1-t**2)])
        p = np.linalg.cholesky(metric(q))@np.random.normal(size=q.shape)
        state = RiemannianLeapfrogState(q, p)
        state.update(riemannnian_auxiliaries)

        step_size = 0.01
        num_steps = 1
        # This check may fail for low precision because the cached version of
        # the generalized leapfrog integrator uses a predictor step.
        thresh = 1e-13
        max_iters = 10000

        glf_state_a, _ = generalized_leapfrog(state, step_size, num_steps, metric, riemannnian_auxiliaries, thresh, max_iters, False)

        state = VectorFieldLeapfrogState(q, p)
        metric_handler = riemannian_metric_handler(metric)
        glf_state_b, _ = vector_field_leapfrog(state, step_size, num_steps, log_posterior, metric_handler, velocity_vector, force_vector, thresh, max_iters)

        self.assertTrue(np.allclose(glf_state_a.position, glf_state_b.position))
        self.assertTrue(np.allclose(glf_state_a.momentum, glf_state_b.momentum))
        self.assertTrue(np.allclose(glf_state_a.log_posterior, glf_state_b.log_posterior))
