from typing import Optional

import numpy as np
import scipy.linalg as spla


def solve_psd(A: np.ndarray, b: Optional[np.ndarray]=None, return_chol: bool=False):
    """Solve the system `A x = b` under the assumption that `A` is positive
    definite. The method implemented is to compute the Cholesky factorization
    of `A` and solve the system via forward-backward substitution.

    Args:
        A: Left-hand side of the linear system.
        b: Right-hand side of the linear system.
        return_chol: Whether or not to return the computed Cholesky factor.

    Returns:
        x: Solution of the linear system.
        L: The Cholesky factor of the left-hand side of the linear system.

    """
    if b is None:
        b = np.eye(len(A))
    L = np.linalg.cholesky(A)
    x = spla.cho_solve((L, True), b)
    if return_chol:
        return x, L
    else:
        return x
