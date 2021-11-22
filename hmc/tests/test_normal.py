import unittest

import numpy as np
import scipy.stats as spst

from hmc.linalg import solve_psd
from hmc.statistics.normal import logpdf


class TestNormal(unittest.TestCase):
    def test_log_density(self):
        n = int(np.ceil(20*np.random.uniform()))
        s = np.random.normal(size=(n, n))
        s = np.matmul(s.T, s)
        m = np.zeros(n)
        x = spst.multivariate_normal.rvs(m, s)
        x = np.atleast_1d(x)
        _, logdet = np.linalg.slogdet(s)
        inv = solve_psd(s)
        self.assertTrue(
            np.allclose(spst.multivariate_normal.logpdf(x, m, s),
                        logpdf(x, logdet, inv)))
