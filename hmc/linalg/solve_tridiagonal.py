from typing import Optional

import numpy as np
import scipy.linalg as spla


def solve_tridiagonal(tri: np.ndarray, rhs: Optional[np.ndarray]=None) -> np.ndarray:
    """The special structure of a tridiagonal matrix permits it to be used in
    solving a linear system in linear time instead of the usual cubic time.

    Args:
        tri: Tridiagonal matrix.
        rhs: Right-hand side of the linear system.

    Returns:
        out: The solution of the linear system involving a tridiagonal matrix.

    """
    if rhs is None:
        rhs = np.eye(tri.shape[-1])
    # ab = np.array([
    #     np.hstack((0.0, np.diag(tri, 1))),
    #     np.diag(tri, 0)
    # ])
    L = spla.cholesky_banded(tri)
    return spla.cho_solve_banded((L, False), rhs)
