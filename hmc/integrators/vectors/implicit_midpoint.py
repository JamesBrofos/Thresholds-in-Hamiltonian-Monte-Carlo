import copy
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import ImplicitMidpointInfo
from hmc.integrators.terminal import cond
from hmc.linalg import solve_psd
from hmc.integrators.fields import riemannian, softabs
from hmc.integrators.states.implicit_midpoint_state import ImplicitMidpointState


def vector_field_from_euclidean_auxiliaries(auxiliaries: Callable, metric: np.ndarray) -> Callable:
    """Computes the velocity and force for a Euclidean Hamiltonian system. The
    velocity is the inverse metric multiplied into the momentum and the force
    is the gradient og the log-posterior.

    Args:
        auxiliaries: Function to compute the log-posterior and the gradient of
            the log-posterior.
        metric: The constant Euclidean metric.

    Returns:
        vector_field: A function returning the time derivatives of position and
            momentum.

    """
    iG = solve_psd(metric)
    def vector_field(q, p):
        lp, glp = auxiliaries(q)
        vel = iG@p
        force = glp
        return vel, force
    return vector_field

def vector_field_from_riemannian_auxiliaries(auxiliaries: Callable) -> Callable:
    """Computes the velocity and force, which are the time derivatives of position
    and momentum, given the auxiliaries of a Riemannian vector field, which
    includes the gradient of the log-posterior, the Riemannian metric, and the
    Jacobian of the Riemannian metric.

    Args:
        auxiliaries: Function to compute the gradient of the log-posterior, the
            Riemannian metric, and the Jacobian of the Riemannian metric.

    Returns:
        vector_field: A function returning the time derivatives of position and
            momentum.

    """
    def vector_field(q, p):
        num_dims = len(q)
        lp, glp, metric, jac_metric = auxiliaries(q)
        jac_metric = np.swapaxes(jac_metric, 0, -1)
        inv_metric = solve_psd(metric)
        dld = riemannian.grad_logdet(inv_metric, jac_metric, num_dims)
        vel = riemannian.velocity(inv_metric, p)
        force = riemannian.force(vel, glp, jac_metric, dld)
        return vel, force
    return vector_field

def vector_field_from_softabs_auxiliaries(auxiliaries: Callable, alpha: float) -> Callable:
    """Computes the velocity and force, which are the time derivatives of position
    and momentum, given the auxiliaries of a SoftAbs vector field, which
    includes the gradient of the log-posterior, the Hessian of the
    log-posterior, and the Jacobian of the Hessian.

    Args:
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Hessian, and the Jacobian of the Hessian.
        alpha: SoftAbs sharpness parameter.

    Returns:
        vector_field: A function returning the time derivatives of position and
            momentum.

    """
    def vector_field(q, p):
        _, glp, H, dH = auxiliaries(q)
        l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, alpha)
        vel = riemannian.velocity(inv_metric, p)
        force = softabs.force(p, glp, dH, l, lt, inv_lt, U, alpha)
        return vel, force
    return vector_field

def midpoint_step(val: Tuple, zo: np.ndarray, half_step: float, vector_field: Callable) -> Tuple:
    """Single step of the implicit midpoint integrator. Computes the midpoint,
    evaluates the gradient at the midpoint, takes a step from the initial
    position in the direction of the gradient at the midpoint, and measures the
    difference between the resulting point and the candidate stationary point.

    """
    zmcand, _, _, num_iters = val
    dz = np.hstack(vector_field(*np.split(zmcand, 2)))
    zm = zo + half_step*dz
    delta = zm - zmcand
    num_iters += 1
    return zm, dz, delta, num_iters

def single_step(
        vector_field: Callable,
        state: ImplicitMidpointState,
        info: ImplicitMidpointInfo,
        step_size: float,
        thresh: float,
        max_iters: int) -> Tuple[ImplicitMidpointState, ImplicitMidpointInfo]:
    """Implements a single step of the implicit midpoint integrator. The implicit
    midpoint integrator is symmetric, symplectic, and second-order accurate
    (third-order local error).

    Args:
        vector_field: The Hamiltonian vector field.
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
    # Initial candidate.
    zo = np.hstack((state.position, state.momentum))
    # Fixed point iteration and half step-size.
    delta = np.ones_like(zo) * np.inf
    half_step = 0.5*step_size
    val = (zo, np.zeros_like(zo), delta, 0)
    while cond(val, thresh, max_iters):
        val = midpoint_step(val, zo, half_step, vector_field)

    # Determine whether or not the integration was successful.
    zm, dzm, delta, num_iters = val
    success = np.max(np.abs(delta)) < thresh
    # Final explicit Euler step.
    zn = zm + half_step * dzm
    state.position, state.momentum = np.split(zn, 2)
    info.num_iters += num_iters
    info.success &= success
    return state, info

def implicit_midpoint(
        state: ImplicitMidpointState,
        step_size: float,
        num_steps: int,
        log_posterior: Callable,
        metric_handler: Callable,
        vector_field: Callable,
        thresh: float,
        max_iters: int
) -> Tuple[ImplicitMidpointState, ImplicitMidpointInfo]:
    """Implements the multiple-step implicit midpoint integrator for computing
    approximate solutions along a vector field.

    Args:
        state: An object containing the current position in phase space.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        log_posterior: The log-density of the posterior from which to sample.
        metric_handler: Function to compute the Riemannian metric, its inverse,
            its matrix square root, and its log-determinant.
        vector_field: The vector field along which to compute the solution of the
            equations of motion.
        thresh: The convergence threshold for fixed point iterations.
        max_iters: The maximum number of fixed point iterations to attempt.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.deepcopy(state)
    info = ImplicitMidpointInfo()
    for i in range(num_steps):
        state, info = single_step(vector_field, state, info, step_size, thresh, max_iters)

    state.log_posterior = log_posterior(state.position)
    metric_handler(state)
    return state, info
