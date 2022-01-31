import os
import pickle

import numpy as np

from hmc.applications import t



def generate_samples(scale, num_samples):
    dof = 5
    Sigma = np.ones(20)
    Sigma[-1] = scale
    Sigma = np.diag(Sigma)
    L = np.linalg.cholesky(Sigma)
    iid = np.array([
        np.hstack(t.sample(L, dof)) for _ in range(num_samples)
    ])
    return iid, Sigma, dof

def main(num_samples, it):
    for scale in [1, 10, 100, 1000, 10000]:
        iid, Sigma, dof = generate_samples(scale, num_samples)
        with open(os.path.join('data', 'samples-{}-scale-{}.pkl'.format(it, scale)), 'wb') as f:
            pickle.dump({'iid': iid, 'Sigma': Sigma, 'dof': dof}, f)


if __name__ == '__main__':
    np.random.seed(42)
    main(1000000, 'all')
    for i in range(10):
        main(1000000, i+1)
