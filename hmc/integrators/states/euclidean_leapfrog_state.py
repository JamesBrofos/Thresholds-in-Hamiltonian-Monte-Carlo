import numpy as np

from .leapfrog_state import LeapfrogState


class EuclideanLeapfrogState(LeapfrogState):
    """The Euclidean leapfrog state implements the state object for Hamiltonian
    Monte Carlo with a constant metric. The Euclidean state needs to be updated
    if either the log-posterior or the gradient of the log-posterior are not
    available.

    """
    @property
    def requires_update(self) -> bool:
        o = self.log_posterior is None or \
            self.grad_log_posterior is None
        return o

    def update(self, auxiliaries):
        log_posterior, grad_log_posterior = auxiliaries(self.position)
        self.log_posterior = log_posterior
        self.grad_log_posterior = grad_log_posterior
        self.velocity = self.inv_metric.dot(self.momentum)
        self.force = self.grad_log_posterior
