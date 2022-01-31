import os
import pickle

import numpy as np

from hmc.applications import neal_funnel


def main(num_samples=1000000, fname='samples.pkl'):
    num_dims = 10
    iid = np.array([
        np.hstack(neal_funnel.sample(num_dims)) for _ in range(num_samples)
    ])
    with open(os.path.join('data', fname), 'wb') as f:
        pickle.dump({'iid': iid, 'num_dims': num_dims}, f)


if __name__ == '__main__':
    np.random.seed(42)
    main()
    for i in range(10):
        main(1000000, 'samples-{}.pkl'.format(i+1))
