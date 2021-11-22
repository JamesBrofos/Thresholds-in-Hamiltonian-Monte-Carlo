import os
import pickle

import numpy as np

from hmc.applications import banana


def generate_data():
    np.random.seed(0)
    t = 0.5
    sigma_y = 2.0
    sigma_theta = 2.0
    num_obs = 100
    theta, y = banana.generate_data(t, sigma_y, sigma_theta, num_obs)
    return y, sigma_y, sigma_theta

def main():
    y, sigma_y, sigma_theta = generate_data()
    with open(os.path.join('data', 'data.pkl'), 'wb') as f:
        pickle.dump({'y': y, 'sigma_y': sigma_y, 'sigma_theta': sigma_theta}, f)


if __name__ == '__main__':
    main()
