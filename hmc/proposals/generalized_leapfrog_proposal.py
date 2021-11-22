from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import GeneralizedLeapfrogInfo
from hmc.integrators.stateful import generalized_leapfrog
from hmc.integrators.states import RiemannianLeapfrogState
from hmc.proposals.proposal import Diagnostics, Proposal, ProposalInfo, error_intercept, momentum_negation


class RiemannianLeapfrogProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters_pos = Diagnostics()
        self.num_iters_mom = Diagnostics()

class RiemannianLeapfrogProposal(Proposal):
    """The Riemannian Hamiltonian Monte Carlo algorithm takes advantage of local
    geometric information in order to adapt proposals to directions in which
    the posterior exhibits the greatest variation locally.

    Parameters:
        metric: Function to compute the Fisher information matrix.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Fisher information matrix, and the Jacobian of the
            Fisher information matrix.
        thresh: The convergence threshold for fixed point iterations.
        max_iters: The maximum number of fixed point iterations to attempt.
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

    """
    def __init__(self,
                 metric: Callable,
                 auxiliaries: Callable,
                 thresh: float,
                 max_iters: int,
                 newton_momentum: bool=False,
                 newton_position: bool=False
    ):
        super().__init__(RiemannianLeapfrogProposalInfo())
        self.metric = metric
        self.auxiliaries = auxiliaries
        self.thresh = thresh
        self.max_iters = max_iters
        self.newton_momentum = newton_momentum
        self.newton_position = newton_position

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: RiemannianLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[RiemannianLeapfrogState, GeneralizedLeapfrogInfo]:
        state, info = generalized_leapfrog(
            state,
            step_size,
            num_steps,
            self.metric,
            self.auxiliaries,
            self.thresh,
            self.max_iters,
            self.newton_momentum,
            self.newton_position
        )
        self.info.num_iters_pos.update(info.num_iters_pos / num_steps)
        self.info.num_iters_mom.update(info.num_iters_mom / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> RiemannianLeapfrogState:
        p = np.zeros_like(qt)
        state = RiemannianLeapfrogState(qt, p)
        state.update(self.auxiliaries)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
