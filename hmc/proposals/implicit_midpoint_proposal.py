from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import ImplicitMidpointInfo
from hmc.integrators.vectors import implicit_midpoint
from hmc.integrators.states import ImplicitMidpointState
from hmc.linalg import solve_psd
from hmc.proposals.proposal import Diagnostics, Proposal, ProposalInfo, error_intercept, momentum_negation


class ImplicitMidpointProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters = Diagnostics()

class ImplicitMidpointProposal(Proposal):
    """The implicit midpoint integrator is an alternative to the generalized
    leapfrog which has better stability and energy conservation properties. The
    disadvantage is that the implicit midpoint integrator has fewer
    opportunities to cache repeated computations.

    Parameters:
        log_posterior: The log-density of the posterior from which to sample.
        metric_handler: Function to compute the Riemannian metric, its inverse,
            its matrix square root, and its log-determinant.
        vector_field: The vector field along which to compute the solution of the
            equations of motion.
        thresh: The convergence threshold for fixed point iterations.
        max_iters: The maximum number of fixed point iterations to attempt.

    """
    def __init__(self, log_posterior: Callable, metric_handler: Callable, vector_field: Callable, thresh: float, max_iters: int):
        super().__init__(ImplicitMidpointProposalInfo())
        self.log_posterior = log_posterior
        self.metric_handler = metric_handler
        self.vector_field = vector_field
        self.thresh = thresh
        self.max_iters = max_iters

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: ImplicitMidpointState,
            step_size: float,
            num_steps: int
    ) -> Tuple[ImplicitMidpointState, ImplicitMidpointInfo]:
        state, info = implicit_midpoint(
            state,
            step_size,
            num_steps,
            self.log_posterior,
            self.metric_handler,
            self.vector_field,
            self.thresh,
            self.max_iters
        )
        return state, info

    def first_state(self, qt: np.ndarray) -> ImplicitMidpointState:
        p = np.zeros_like(qt)
        state = ImplicitMidpointState(qt, p)
        state.log_posterior = self.log_posterior(state.position)
        self.metric_handler(state)
        return state
