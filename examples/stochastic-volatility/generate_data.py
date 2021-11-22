import os
import pickle

import numpy as np

from hmc.applications import stochastic_volatility


def generate_data():
    np.random.seed(0)
    T = 1000
    sigma = 0.15
    phi = 0.98
    beta = 0.65
    x, y = stochastic_volatility.generate_data(T, sigma, phi, beta)
    return x, y, sigma, phi, beta

def main():
    x, y, sigma, phi, beta = generate_data()
    with open(os.path.join('data', 'data.pkl'), 'wb') as f:
        pickle.dump({'x': x, 'y': y, 'sigma': sigma, 'phi': phi, 'beta': beta}, f)

if __name__ == '__main__':
    main()
