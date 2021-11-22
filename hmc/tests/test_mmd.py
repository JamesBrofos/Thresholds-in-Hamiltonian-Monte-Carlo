import unittest

import numpy as np
from scipy.spatial.distance import cdist

from hmc import mmd

class TestMMD(unittest.TestCase):
    def test_mmd(self):
        n = int(1000*np.random.uniform())
        m = int(1000*np.random.uniform())
        k = int(10*np.random.uniform())
        x = np.random.normal(size=(m, k))
        y = np.random.normal(size=(n, k))
        bw = np.random.exponential()
        u = mmd(x, y, bw)

        Kxx = np.exp(-0.5*cdist(x, x, 'sqeuclidean') / bw**2)
        Kyy = np.exp(-0.5*cdist(y, y, 'sqeuclidean') / bw**2)
        Kxy = np.exp(-0.5*cdist(x, y, 'sqeuclidean') / bw**2)
        a = 2*np.sum(Kxx[np.triu_indices(m, 1)]) / (m*(m-1))
        b = 2*np.sum(Kyy[np.triu_indices(n, 1)]) / (n*(n-1))
        c = -2*np.sum(Kxy) / (m*n)
        v = a + b + c
        self.assertTrue(np.allclose(u, v))
