import os
import pickle

import numpy as np

from hmc.applications.fitzhugh_nagumo import generate_data


def main():
    np.random.seed(0)
    state = np.array([1.0, -1.0])
    t = np.linspace(0.0, 10.0, 200)
    sigma = 0.5
    y = generate_data(state, t, sigma)
    with open(os.path.join('data', 'data.pkl'), 'wb') as f:
        pickle.dump({'y': y, 't': t, 'sigma': sigma, 'state': state}, f)

if __name__ == '__main__':
    main()
