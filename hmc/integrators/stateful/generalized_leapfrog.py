import copy
from typing import Callable, Tuple

import numpy as np
import scipy.linalg as spla

from hmc.integrators.info import GeneralizedLeapfrogInfo
from hmc.integrators.states import RiemannianLeapfrogState
from hmc.integrators.terminal import cond
from hmc.integrators.fields import riemannian
from hmc.linalg import solve_psd


def momentum_step(
        val: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
        half_step: float,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the momentum variable.

    Args:
        val: Tuple containing the fixed point momentum and velocity, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        half_step: One-half the integration step-size.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
         pm: The updated momentum.
         vm: The updated velocity.
         delta: The iteration-over-iteration difference in the momentum.
         num_iters: The number of fixed point iterations incremented by one.

    """
    pmcand, _, _, num_iters = val
    # Compute the gradient of the Hamiltonian with respect to position.
    vm = riemannian.velocity(state.inv_metric, pmcand)
    f = riemannian.force(vm, state.grad_log_posterior, state.jac_metric, state.grad_logdet_metric)
    pm = state.momentum + half_step * f
    delta = pm - pmcand
    num_iters += 1
    return pm, vm, delta, num_iters

def newton_momentum_step(
        val: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
        o: np.ndarray,
        E: np.ndarray,
        Id: np.ndarray,
        half_step: float,
        state: RiemannianLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Updates the momentum variable by finding the solution to a root-finding
    problem via Newton's method. Newton's method has a faster order of
    convergence than fixed point iteration but has a greater computational
    burden, since it involves inverting the Jacobian of the function whose root
    is sought.

    """
    pmcand, _, num_iters = val
    po = state.momentum
    f = o + 0.5*pmcand@E@pmcand
    g = pmcand - (po + half_step*f)
    J = Id - half_step*E@pmcand
    x = np.linalg.solve(J, g)
    pm = pmcand - x
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        metric: Callable,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the position variable.

    Args:
        val: Tuple containing the fixed point position, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        half_step: One-half the integration step-size.
        metric: Function to compute the Fisher information metric.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
        qn: The updated position variable.
        delta: The iteration-over-iteration difference in the position.
        num_iters: The number of fixed point iterations incremented by one.

    """
    qncand, _, num_iters = val
    newvel = solve_psd(metric(qncand), state.momentum)
    sumvel = state.velocity + newvel
    qn = state.position + half_step * sumvel
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def newton_position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        auxiliaries: Callable,
        Id: np.ndarray,
        state: RiemannianLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    qncand, _, _, num_iters = val
    _, _, G, dG = auxiliaries(qncand)
    iG = solve_psd(G, Id)
    E = np.einsum('ij,jkl->ikl', iG, np.einsum('ijk,jl->ilk', dG, iG))
    Ep = np.einsum('ijk,j->ik', E, state.momentum)
    F = half_step*Ep
    J = Id + F
    g = qncand - state.position - half_step*(iG@state.momentum + state.velocity)
    x = np.linalg.solve(J, g)
    qn = qncand - x
    delta = qn - qncand
    num_iters += 1
    return qn, F, delta, num_iters

def single_step(
        metric: Callable,
        auxiliaries: Callable,
        state: RiemannianLeapfrogState,
        info: GeneralizedLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
        newton_momentum: bool,
        newton_position: bool,
        newton_stability: bool
) -> Tuple[RiemannianLeapfrogState, GeneralizedLeapfrogInfo]:
    """Implements a single step of the generalized leapfrog integrator, which
    involves an implicitly-defined update of the momentum, an implicit update
    to position, and a second explicit update to momentum.

    Args:
        metric: Function to compute the Fisher information metric.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, and the metric, its gradient, and the gradient
            log-determinant of the metric.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step-size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An augmented information object with the updated number of fixed
            point iterations and boolean indicator for successful integration.

    """
    # Precompute the half step-size and the number of dimensions of the
    # position variable. Extract the position and momentum variables.
    half_step = 0.5 * step_size
    qo, po = state.position, state.momentum
    num_dims = len(qo)

    # Precompute the initial difference vector, which is set to be an array of
    # infinite values.
    delta = np.inf*np.ones(num_dims)
    if state.requires_update:
        raise ValueError('State was not prepared with `update`.')

    # The first step of the integrator is to find a fixed point of the momentum
    # variable.
    if not newton_momentum:
        val = (po + half_step*state.force, delta, delta, 0)
        while cond(val, thresh, max_iters):
            val = momentum_step(val, half_step, state)
        pm, vm, delta_mom, num_iters_mom = val
        success_mom = np.max(np.abs(delta_mom)) < thresh
    else:
        val = (po + half_step*state.force, delta, 0)
        Id = np.eye(num_dims)
        E = state.inv_metric@state.jac_metric@state.inv_metric
        o = state.grad_log_posterior - state.grad_logdet_metric
        while cond(val, thresh, max_iters):
            val = newton_momentum_step(val, o, E, Id, half_step, state)
        pm, delta_mom, num_iters_mom = val
        vm = riemannian.velocity(state.inv_metric, pm)
        J = half_step*E@pm
        success_mom = np.max(np.abs(delta_mom)) < thresh
        # Check if the computed fixed point solution is locally stable. This is
        # equivalent to the operator norm of the Jacobian being less than one.
        if newton_stability:
            _, s, _ = np.linalg.svd(J)
            if np.max(s) >= 1.0:
                success_mom = False

    state.velocity = vm
    state.momentum = pm

    # The second step of the integrator is to find a fixed point of the
    # position variable. The first momentum gradient could be conceivably
    # cached and saved.
    if not newton_position:
        val = (qo + step_size*state.velocity, delta, 0)
        while cond(val, thresh, max_iters):
            val = position_step(val, half_step, metric, state)
        qn, delta_pos, num_iters_pos = val
        success_pos = np.max(np.abs(delta_pos)) < thresh
    else:
        Id = np.eye(num_dims)
        val = (qo + step_size*state.velocity, Id, delta, 0)
        while cond(val, thresh, max_iters):
            val = newton_position_step(val, half_step, auxiliaries, Id, state)
        qn, J, delta_pos, num_iters_pos = val
        success_pos = np.max(np.abs(delta_pos)) < thresh
        if newton_stability:
            _, s, _ = np.linalg.svd(J)
            if np.max(s) >= 1.0:
                success_pos = False

    # Last step is to do an explicit half-step of the momentum variable.
    state.position = qn
    state.update(auxiliaries)
    state.momentum += half_step*state.force

    # Determine if integration was successful (to the desired precision).
    success = np.logical_and(success_mom, success_pos)
    # Update the information on the current state by incrementing the total
    # number of iterations and whether or not the most recent step had
    # successful integration.
    info.num_iters_pos += num_iters_pos
    info.num_iters_mom += num_iters_mom
    info.success &= success
    return state, info

def generalized_leapfrog(
        state: RiemannianLeapfrogState,
        step_size: float,
        num_steps: int,
        metric: Callable,
        auxiliaries: Callable,
        thresh: float,
        max_iters: int,
        newton_momentum: bool,
        newton_position: bool,
        newton_stability: bool
) -> Tuple[RiemannianLeapfrogState, GeneralizedLeapfrogInfo]:
    """Implements the generalized leapfrog integrator which avoids recomputing
    redundant quantities at each iteration.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        metric: Function to compute the Fisher information metric.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, and the metric, its gradient, and the gradient
            log-determinant of the metric.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.deepcopy(state)
    info = GeneralizedLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(
            metric,
            auxiliaries,
            state,
            info,
            step_size,
            thresh,
            max_iters,
            newton_momentum,
            newton_position,
            newton_stability
        )

    state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
    return state, info
