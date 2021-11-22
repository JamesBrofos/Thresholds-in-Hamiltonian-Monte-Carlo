import os
import pickle

import numpy as np

from hmc.applications import neal_funnel


def generate_samples():
    np.random.seed(42)
    num_dims = 10
    iid = np.array([
        np.hstack(neal_funnel.sample(num_dims)) for _ in range(1000000)
    ])
    return iid, num_dims

def main():
    iid, num_dims = generate_samples()
    with open(os.path.join('data', 'samples.pkl'), 'wb') as f:
        pickle.dump({'iid': iid, 'num_dims': num_dims}, f)


if __name__ == '__main__':
    main()
