from typing import Callable, Tuple

import numpy as np
import scipy.sparse as spsr

from hmc.integrators.info import Info
from hmc.integrators.stateful import leapfrog
from hmc.integrators.states import EuclideanLeapfrogState
from hmc.linalg import solve_psd
from hmc.proposals.proposal import ProposalInfo, Proposal, error_intercept, momentum_negation


class EuclideanLeapfrogProposal(Proposal):
    """The leapfrog integrator assumes a fixed metric, which is supplied to the
    constructor of the leapfrog proposal. The required quantities of the metric
    are then computed and cached.

    Args:
        auxiliaries: Function to compute the log-posterior and the gradient of
            the log-posterior.
        metric: A positive definite matrix.

    """
    def __init__(self, auxiliaries: Callable, metric: np.ndarray, solve=solve_psd, chol=np.linalg.cholesky):
        super().__init__(ProposalInfo())
        self.auxiliaries = auxiliaries
        if metric.ndim == 1:
            self.metric = metric
            self.inv_metric = spsr.diags(np.reciprocal(metric))
            self.sqrtm_metric = spsr.diags(np.sqrt(metric))
        else:
            self.metric = metric
            self.inv_metric = solve(self.metric)
            self.sqrtm_metric = chol(self.metric)
        if spsr.issparse(self.sqrtm_metric):
            self.logdet_metric = 2.0*np.sum(np.log(self.sqrtm_metric.diagonal()))
        else:
            self.logdet_metric = 2.0*np.sum(np.log(np.diag(self.sqrtm_metric)))

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: EuclideanLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[EuclideanLeapfrogState, Info]:
        state, info = leapfrog(state, step_size, num_steps, self.auxiliaries)
        return state, info

    def first_state(self, qt: np.ndarray) -> EuclideanLeapfrogState:
        p = np.zeros_like(qt)
        state = EuclideanLeapfrogState(qt, p)
        state.metric = self.metric
        state.inv_metric = self.inv_metric
        state.sqrtm_metric = self.sqrtm_metric
        state.logdet_metric = self.logdet_metric
        state.update(self.auxiliaries)
        return state
