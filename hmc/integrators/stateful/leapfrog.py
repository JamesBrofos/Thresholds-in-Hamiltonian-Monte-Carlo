import copy
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import LeapfrogInfo
from hmc.integrators.states import EuclideanLeapfrogState


def single_step(
        auxiliaries: Callable,
        state: EuclideanLeapfrogState,
        info: LeapfrogInfo,
        step_size: float
) -> Tuple[EuclideanLeapfrogState, LeapfrogInfo]:
    """Implements a single step of the leapfrog integrator, which is symmetric,
    symplectic, and second-order accurate for separable Hamiltonian systems.

    Args:
        auxiliaries: Function to compute the log-posterior and the gradient of
            the log-posterior.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior
            and gradients.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step_size.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and its gradient.
        info: An information object with the indicator of successful integration.

    """
    if state.requires_update:
        raise ValueError('State was not prepared with `update`.')
    half_step = 0.5*step_size
    state.momentum += half_step * state.force
    state.velocity = state.inv_metric.dot(state.momentum)
    state.position += step_size * state.velocity
    state.update(auxiliaries)
    state.momentum += half_step * state.force
    return state, info

def leapfrog(
        state: EuclideanLeapfrogState,
        step_size: float,
        num_steps: int,
        auxiliaries: Callable
) -> Tuple[EuclideanLeapfrogState, LeapfrogInfo]:
    """Implements a the leapfrog integrator for a separable Hamiltonians.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior
            and gradients.
        step_size: Integration step_size.
        num_steps: Number of integration steos.
        auxiliaries: Function to compute the log-posterior and the gradient of
            the log-posterior.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and its gradient.
        info: An information object with the indicator of successful integration.

    """
    state = copy.deepcopy(state)
    info = LeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(auxiliaries, state, info, step_size)
    return state, info
