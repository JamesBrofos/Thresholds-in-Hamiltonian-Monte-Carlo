from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import GeneralizedLeapfrogInfo
from hmc.integrators.stateful import softabs_leapfrog
from hmc.integrators.states import SoftAbsLeapfrogState
from hmc.proposals.proposal import Proposal, error_intercept, momentum_negation
from hmc.proposals.generalized_leapfrog_proposal import RiemannianLeapfrogProposalInfo


class SoftAbsLeapfrogProposal(Proposal):
    """The Riemannian Hamiltonian Monte Carlo algorithm takes advantage of
    second-order geometry in order to adapt proposals to directions of greatest
    variation in the posterior. The SoftAbs metric is a smooth transformation
    of the Hessian so that it becomes positive definite, giving a generic
    metric compatible with Riemannian manifold HMC.

    Parameters:
        alpha: The SoftAbs sharpness parameter.
        hessian: Function to compute the Hessian of the log-posterior.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Hessian, and the Jacobian of the Hessian.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    """
    def __init__(self, alpha: float, hessian: Callable, auxiliaries: Callable, thresh: float, max_iters: int):
        super().__init__(RiemannianLeapfrogProposalInfo())
        self.alpha = alpha
        self.hessian = hessian
        self.auxiliaries = auxiliaries
        self.thresh = thresh
        self.max_iters = max_iters

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: SoftAbsLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[SoftAbsLeapfrogState, GeneralizedLeapfrogInfo]:
        state, info = softabs_leapfrog(
            state,
            step_size,
            num_steps,
            self.hessian,
            self.auxiliaries,
            self.thresh,
            self.max_iters
        )
        self.info.num_iters_pos.update(info.num_iters_pos / num_steps)
        self.info.num_iters_mom.update(info.num_iters_mom / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> SoftAbsLeapfrogState:
        p = np.zeros_like(qt)
        state = SoftAbsLeapfrogState(qt, p, self.alpha)
        state.update(self.auxiliaries)
        state.sqrtm_metric = np.linalg.cholesky(state.metric)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
