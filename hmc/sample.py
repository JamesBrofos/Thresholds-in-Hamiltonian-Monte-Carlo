import time
from typing import Callable, Tuple

import numpy as np

from hmc.proposals import Proposal, ProposalInfo
from hmc.integrators.info import Info
from hmc.integrators.states import State
from hmc.hamiltonian import hamiltonian
from hmc.statistics.normal import rvs
from hmc.transforms import identity


def metropolis_hastings(
        proposal: Proposal,
        state: State,
        step_size: float,
        ns: int,
        unif: float,
        inverse_transform: Callable
) -> Tuple[State, Info, np.ndarray, bool]:
    """Computes the Metropolis-Hastings accept-reject criterion given a proposal, a
    current state of the chain, a integration step-size, and a number of
    itnegration steps. We also provide a uniform random variable for
    determining the accept-reject criterion and the inverse transformation
    function for transforming parameters from an unconstrained space to a
    constrained space.

    Args:
        proposal: A proposal operator to advance the state of the Markov chain.
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        step_size: The integration step-size.
        num_steps: The number of integration steps.
        unif: Uniform random number for determining the accept-reject decision.
        inverse_transform: Inverse transformation to map samples back to the
            original space.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.
        q: The position variable in the constrained space.
        accept: Whether or not the proposal was accepted.

    """
    ham = hamiltonian(
        state.momentum,
        state.log_posterior,
        state.logdet_metric,
        state.inv_metric)
    q, fldj = inverse_transform(state.position)
    ildj = -fldj
    new_state, prop_info = proposal.propose(state, step_size, ns)
    new_chol, new_logdet = new_state.sqrtm_metric, new_state.logdet_metric
    new_q, new_fldj = inverse_transform(new_state.position)
    new_ham = hamiltonian(
        new_state.momentum,
        new_state.log_posterior,
        new_state.logdet_metric,
        new_state.inv_metric)
    # Notice the relevant choice of sign when the Jacobian determinant of the
    # forward or inverse transform is used.
    #
    # Write this expression as,
    # (exp(-new_ham) / exp(new_fldj)) / (exp(-ham) * exp(ildj))
    #
    # See the following resource for understanding the Metropolis-Hastings
    # correction with a Jacobian determinant correction [1].
    #
    # [1] https://wiki.helsinki.fi/download/attachments/48865399/ch7-rev.pdf
    logu = np.log(unif)
    metropolis = logu < ham - new_ham - new_fldj - ildj + prop_info.logdet
    accept = np.logical_and(metropolis, prop_info.success)
    if accept:
        state = new_state
        q = new_q
        ildj = -new_fldj
    state.momentum *= -1.0
    return state, prop_info, q, accept

def sample(
        q: np.ndarray,
        step_size: float,
        num_steps: int,
        proposal: Proposal,
        forward_transform: Callable=identity,
        inverse_transform: Callable=identity,
        partial_momentum: float=0.0,
        check_prob: float=0.0,
) -> Tuple[np.ndarray, ProposalInfo]:
    """Draw samples from the target density using Hamiltonian Monte Carlo. This
    function requires that one specify a Hamiltonian energy, a proposal
    operator, and a function to sample momenta. This function is implemented as
    a generator so as to yield samples from the target distribution when
    requested.

    Args:
        q: The position variable.
        step_size: The integration step-size.
        num_steps: The number of integration steps.
        proposal: A proposal operator to advance the state of the Markov chain.
        forward_transform: Transforms the state space to make sampling easier.
        inverse_transform: Inverse transformation to map samples back to the
            original space.
        partial_momentum: Parameter controlling the partial refreshment of the
            momentum variable.
        check_prob: Probability to compute reversibility and volume preservation
            statistics for the proposal.

    Returns:
        q: The next position variable.
        info: Diagnostic information about the transition.
        elapsed: The time to compute the next state of the chain (excluding
            computation time for the metrics).

    """
    # Transform the position variable if, for instance, an unconstrained
    # representation is required.
    qt, ildj = forward_transform(q)
    state = proposal.first_state(qt)
    del qt
    alpha = partial_momentum
    beta = np.sqrt(1-alpha**2)

    # Iteration counter for number of sampling steps and acceptance rates and
    # average number of internal integrator iterations.
    siter = 0

    while True:
        # Increment the number of sampling steps.
        siter += 1
        # Keep track of time to complete the sampling step but not to compute
        # checks of reversibility and detailed balance.
        start = time.time()

        # Sample momentum from conditional distribution and compute the
        # associated Hamiltonian energy.
        p = state.momentum
        n = rvs(state.sqrtm_metric)
        state.momentum = alpha*p + beta*n
        # Randomize the number of integration steps.
        ns = int(np.ceil(np.random.uniform() * num_steps))
        unif = np.random.uniform()
        new_state, prop_info, q, accept = metropolis_hastings(
            proposal, state, step_size, ns, unif, inverse_transform)
        elapsed = time.time() - start

        # Randomly check the properties of reversibility and volume
        # preservation.
        random_check = np.random.uniform() < check_prob
        if random_check and not prop_info.invalid:
            proposal.random_check(state, step_size, ns, prop_info)

        proposal.update_accept(accept)
        info = proposal.info
        state = new_state
        yield q, info, elapsed
