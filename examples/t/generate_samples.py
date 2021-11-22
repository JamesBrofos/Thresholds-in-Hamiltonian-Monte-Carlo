import os
import pickle

import numpy as np

from hmc.applications import t


def generate_samples(scale):
    np.random.seed(42)
    dof = 5
    Sigma = np.ones(20)
    Sigma[-1] = scale
    Sigma = np.diag(Sigma)
    L = np.linalg.cholesky(Sigma)
    iid = np.array([
        np.hstack(t.sample(L, dof)) for _ in range(1000000)
    ])
    return iid, Sigma, dof

def main():
    for scale in [1, 10, 100, 1000, 10000]:
        iid, Sigma, dof = generate_samples(scale)
        with open(os.path.join('data', 'samples-scale-{}.pkl'.format(scale)), 'wb') as f:
            pickle.dump({'iid': iid, 'Sigma': Sigma, 'dof': dof}, f)


if __name__ == '__main__':
    main()
