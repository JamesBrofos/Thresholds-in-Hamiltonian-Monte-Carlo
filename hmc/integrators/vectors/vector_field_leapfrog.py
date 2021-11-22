import copy
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import GeneralizedLeapfrogInfo
from hmc.integrators.terminal import cond
from hmc.linalg import solve_psd
from hmc.integrators.fields import riemannian, softabs
from hmc.integrators.states.vector_field_leapfrog_state import VectorFieldLeapfrogState


def velocity_and_force_from_riemannian_auxiliaries(metric: Callable, auxiliaries: Callable) -> Tuple[Callable]:
    """Function to produce functions for computing the velocity and force vector
    fields given a Riemannian Hamiltonian system.

    Args:
        metric: Function to compute the Riemannian metric.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Riemannian metric, and the gradient of the
            Riemannian metric.

    Returns:
        velocity_vector: Function computing the time derivative of position.
        force_vector: Function computing the time derivative of momentum.

    """
    def velocity_vector(q, p):
        G = metric(q)
        vel = riemannian.velocity(solve_psd(G), p)
        return vel

    def force_vector(q, p):
        num_dims = len(q)
        lp, glp, G, dG = auxiliaries(q)
        jac_metric = np.swapaxes(dG, 0, -1)
        inv_metric = solve_psd(G)
        grad_logdet_metric = riemannian.grad_logdet(inv_metric, jac_metric, num_dims)
        vel = inv_metric@p
        force = riemannian.force(vel, glp, jac_metric, grad_logdet_metric)
        return force
    return velocity_vector, force_vector

def velocity_and_force_from_softabs_auxiliaries(hessian: Callable, auxiliaries: Callable, alpha: float) -> Tuple[Callable]:
    """Function to produce functions for computing the velocity and force vector
    fields given a SoftAbs Hamiltonian system.

    Args:
        hessian: Function to compute the Hessian of the log-posterior.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Hessian, and the Jacobian of the Hessian.
        alpha: SoftAbs sharpness parameter.

    Returns:
        velocity_vector: Function computing the time derivative of position.
        force_vector: Function computing the time derivative of momentum.

    """
    def velocity_vector(q, p):
        H = hessian(q)
        l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, alpha)
        vel = riemannian.velocity(inv_metric, p)
        return vel

    def force_vector(q, p):
        _, glp, H, dH = auxiliaries(q)
        l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, alpha)
        force = softabs.force(p, glp, dH, l, lt, inv_lt, U, alpha)
        return force
    return velocity_vector, force_vector

def momentum_step(val: Tuple, qo: np.ndarray, po: np.ndarray, half_step: float, force_vector: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the momentum variable."""
    pmcand, _, num_iters = val
    dp = force_vector(qo, pmcand)
    pm = po + half_step*dp
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(val: Tuple, qo: np.ndarray, po: np.ndarray, half_step: float, velocity_vector: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the position variable."""
    qncand, _, num_iters = val
    dq = velocity_vector(qo, po) + velocity_vector(qncand, po)
    qn = qo + half_step*dq
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def single_step(
        velocity_vector: Callable,
        force_vector: Callable,
        state: VectorFieldLeapfrogState,
        info: GeneralizedLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int
) -> Tuple[VectorFieldLeapfrogState, GeneralizedLeapfrogInfo]:
    """Implements a single step of the generalized leapfrog integrator to compute a
    trajectory of a non-separable Hamiltonian. This implementation of the
    generalized leapfrog integrator does not incorporate caching.

    Args:
        velocity_vector: Vector field for the rate of change of the position
            variable.
        force_vector: Vector field for the rate of change of the momentum
            variable.
        state: An object containing the position and momentum variables of the
            state in phase space.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step_size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    # Fixed point iteration and half step-size.
    qo, po = state.position, state.momentum
    delta = np.ones_like(state.position) * np.inf
    half_step = 0.5*step_size
    # Fixed point iteration to determine the first half-step in the momentum.
    val = (po, delta, 0)
    while cond(val, thresh, max_iters):
        val = momentum_step(val, qo, po, half_step, force_vector)
    pm, delta_mom, num_iters_mom = val
    success_mom = np.max(np.abs(delta_mom)) < thresh

    # Fixed point iteration to determine the next position.
    val = (qo, delta, 0)
    while cond(val, thresh, max_iters):
        val = position_step(val, qo, pm, half_step, velocity_vector)
    qn, delta_pos, num_iters_pos = val
    success_pos = np.max(np.abs(delta_pos)) < thresh

    # Final update to the momentum variable.
    pn = pm + half_step*force_vector(qn, pm)
    state.position, state.momentum = qn, pn
    info.num_iters_pos += num_iters_pos
    info.num_iters_mom += num_iters_mom
    info.success &= np.logical_and(success_pos, success_mom)
    return state, info

def vector_field_leapfrog(
        state: VectorFieldLeapfrogState,
        step_size: float,
        num_steps: int,
        log_posterior: Callable,
        metric_handler: Callable,
        velocity_vector: Callable,
        force_vector: Callable,
        thresh: float,
        max_iters: int
) -> Tuple[VectorFieldLeapfrogState, GeneralizedLeapfrogInfo]:
    """Implements the multiple-step generalized leapfrog integrator (without
    caching) for computing proposals for use in Hamiltonian Monte Carlo.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space.
        step_size: Integration step_size.
        num_steps: Number of integration steps.
        log_posterior: The log-density of the posterior from which to sample.
        metric_handler: Function to compute the Riemannian metric, its inverse,
            its matrix square root, and its log-determinant.
        velocity_vector: Vector field for the rate of change of the position
            variable.
        force_vector: Vector field for the rate of change of the momentum
            variable.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.deepcopy(state)
    info = GeneralizedLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(velocity_vector, force_vector, state, info, step_size, thresh, max_iters)

    state.log_posterior = log_posterior(state.position)
    metric_handler(state)
    return state, info
