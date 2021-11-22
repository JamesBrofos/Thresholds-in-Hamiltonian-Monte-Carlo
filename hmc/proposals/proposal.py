import abc
import copy
import functools
from typing import Callable, Tuple

import numpy as np

from hmc.integrators.info import Info
from hmc.integrators.states import (
    State, ImplicitMidpointState, VectorFieldLeapfrogState)

class Diagnostics:
    """Convenience class for working with averages and summations of diagnostic
    quantities. Also keeps a list of all recorded values.

    Parameters:
        values: List of values.
        avg: The average of the list of valus.
        summ: The sum of the list of values.
        num_values: The number of values used in computing the sum or average.

    """
    def __init__(self):
        self.values: list = []
        self.avg: float = np.nan
        self.summ: float = np.nan
        self.num_values: int = 0

    def update(self, val: float):
        self.values.append(val)
        if not np.isnan(val) and not np.isinf(val):
            self.num_values += 1
            if self.num_values == 1:
                self.summ = val
            else:
                self.summ += val
            self.avg = self.summ / self.num_values

class ProposalInfo:
    """Diagnostic information from running the Hamiltonian Monte Carlo sampling
    procedure.

    Parameters:
        accept: The acceptance probability of the Markov chain.
        num_iters: The number of internal iterations computed by the numerical
            integrator.
        absrev: The absolute error in the reversibility of the integrator.
        relrev: The relative error in the reversibility of the integrator.
        jacdet: The error in the volume preservation of the integrator as
            measured by its difference from a unit Jacobian.
        invalid: Counter of how many invalid proposals have been generated.

    """
    def __init__(self):
        self.accept = Diagnostics()
        self.absrev = Diagnostics()
        self.relrev = Diagnostics()
        self.jacdet = {
            _: Diagnostics() for _ in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        }
        self.invalid = Diagnostics()

def error_intercept(proposal):
    """Intercepts proposals to check for error states and to intercept them before
    they propagate any further.

    Args:
        proposal: Function to generate a proposal given a current state, a number
            of integration steps, and an integration step-size.

    Returns:
        wrap: The wrapped proposal that intercepts errors.

    """
    @functools.wraps(proposal)
    def wrap(*args, **kwargs):
        try:
            value = proposal(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError):
            state = args[1]
            info = Info()
            info.success = False
            info.invalid = True
            value = (state, info)
        return value
    return wrap

def momentum_negation(proposal):
    @functools.wraps(proposal)
    def wrap(*args, **kwargs):
        value = proposal(*args, **kwargs)
        state, info = value
        state.momentum *= -1.0
        return state, info
    return wrap

class Proposal(abc.ABC):
    """The proposal object provides a generic interface to two methods. The first
    computes a proposal for use in Hamiltonian Monte Carlo by integrating
    Hamilton's equations of motion with a given step-size and a prescribed
    number of steps. The second is a method to generate an initial state from a
    given initial position in phase space.

    """
    def __init__(self, info: ProposalInfo):
        self.info = info

    @abc.abstractmethod
    def propose(
            self,
            state: State,
            step_size: float,
            num_steps: int
    ) -> Tuple[State, Info]:
        raise NotImplementedError()

    @abc.abstractmethod
    def first_state(self, qt: np.ndarray) -> State:
        raise NotImplementedError()

    def random_check(self, state: State, step_size: float, num_steps: int, info):
        for k in self.info.jacdet.keys():
            det = jacobian_determinant(state, step_size, num_steps, self, info.logdet, k)
            self.info.jacdet[k].update(det)
        absrev, relrev = reverse(state, step_size, num_steps, self)
        self.info.absrev.update(absrev)
        self.info.relrev.update(relrev)

    def update_accept(self, accept: bool):
        self.info.accept.update(int(accept))

def reverse(state: State, step_size: float, num_steps: int, proposal: Proposal):
    """Compute the reversibility of the proposal operator by first integrating
    forward, then flipping the sign of the momentum, integrating again, and
    flipping the sign of the momentum a final time in order to compute the
    distance between the original position and the terminal position. If the
    operator is symmetric (reversible) then this distance should be very
    small.

    Args:
        state: The state of the Markov chain.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.

    Returns:
        log_abserr: The absolute error of the original position in phase space and
            the terminal position of the proposal operator.
        log_relerr: The relative error of the original position in phase space and
            the terminal position of the proposal operator.

    """
    q, p = state.position, state.momentum
    sp, info_p = proposal.propose(state, step_size, num_steps)
    sr, info_r = proposal.propose(sp, step_size, num_steps)
    qr, pr = sr.position, sr.momentum
    rev = np.sqrt(np.square(np.linalg.norm(q - qr)) + np.square(np.linalg.norm(p - pr)))
    abserr = np.maximum(rev, 1e-16)
    relerr = abserr / np.sqrt(np.square(np.linalg.norm(q)) + np.square(np.linalg.norm(p)))
    success = info_p.success and info_r.success
    log_abserr = np.log10(abserr) if success else np.nan
    log_relerr = np.log10(relerr) if success else np.nan
    return log_abserr, log_relerr

def jacobian(func: Callable, delta: float):
    """Finite differences approximation to the Jacobian."""
    def jacfn(z):
        num_dims = len(z)
        Jac = np.zeros((num_dims, num_dims))
        for j in range(num_dims):
            pert = np.zeros(num_dims)
            pert[j] = 0.5 * delta
            zh = func(z + pert)
            zl = func(z - pert)
            Jac[j] = (np.hstack(zh) - np.hstack(zl)) / delta
        return Jac
    return jacfn

non_stateful = (ImplicitMidpointState, VectorFieldLeapfrogState)

def jacobian_determinant(
        state: State,
        step_size: float,
        num_steps: int,
        proposal: Proposal,
        logdetact,
        delta: float
) -> float:
    """Compute the Jacobian of the transformation for a single sample consisting of
    a position and momentum.

    Args:
        state: The state of the Markov chain.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        proposal: Proposal operator for the Markov chain; operates on both
            position and momentum.
        delta: Perturbation size for numerical Jacobian.

    Returns:
        err: The logarithm of the error between the Jacobian determinant of the
            transformation computed using finite differences and unity.

    """
    # Redefine the proposal operator as a map purely to phase-space to
    # phase-space with no additional inputs or outputs.
    def _proposal(z):
        q, p = np.split(z, 2)
        s = copy.deepcopy(state)
        s.position = q
        s.momentum = p
        if type(s) not in non_stateful:
            s.update(proposal.auxiliaries)
        prop, info = proposal.propose(s, step_size, num_steps)
        if info.success:
            zp = np.hstack((prop.position, prop.momentum))
        else:
            zp = np.full(z.shape, np.nan)
        return zp

    z = np.hstack((state.position, state.momentum))
    Jac = jacobian(_proposal, delta)(z)
    det = np.abs(np.linalg.det(Jac))
    err = np.maximum(np.abs(det - np.exp(logdetact)), 1e-16)
    err = np.log10(err)
    return err
