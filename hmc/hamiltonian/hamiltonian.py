import numpy as np

from hmc.statistics.normal import logpdf


def hamiltonian(
        momentum: np.ndarray,
        log_posterior: float,
        logdet: float,
        inv: np.ndarray
) -> float:
    """Hamiltonian for sampling from the distribution.

    Args:
        momentum: The momentum variable at which to evaluate the Hamiltonian.
        log_posterior: The value of the log-posterior representing the negative
            potential energy of the system.
        logdet: The log-determinant of the covariance matrix.
        inv: The inverse of the covariance matrix.

    Returns:
        H: The value of the Hamiltonian, representing the total energy of the
            system; this is the sum of the potential and kinetic energies.

    """
    U = -log_posterior
    K = -logpdf(momentum, logdet, inv)
    H = U + K
    return H
