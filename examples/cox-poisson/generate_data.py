import os
import pickle

import numpy as np

from hmc.applications import cox_poisson


def generate_data():
    np.random.seed(0)
    beta = 1.0 / 33
    sigmasq = 1.91
    mu = np.log(126.0) - sigmasq / 2.0
    num_grid = 16
    dist, x, y = cox_poisson.generate_data(num_grid, mu, beta, sigmasq)
    return dist, mu, x, y, sigmasq, beta, num_grid

def main():
    dist, mu, x, y, sigmasq, beta, num_grid = generate_data()
    with open(os.path.join('data', 'data-{}.pkl'.format(num_grid)), 'wb') as f:
        pickle.dump({'dist': dist, 'mu': mu, 'x': x, 'y': y, 'sigmasq': sigmasq, 'beta': beta, 'num_grid': num_grid}, f)

if __name__ == '__main__':
    main()
