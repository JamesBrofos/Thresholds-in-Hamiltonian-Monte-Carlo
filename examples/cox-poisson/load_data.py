import os
import pickle


def load_data(num_grid):
    with open(os.path.join('data', 'data-{}.pkl'.format(num_grid)), 'rb') as f:
        d = pickle.load(f)
        dist = d['dist']
        mu = d['mu']
        x = d['x']
        y = d['y']
        sigmasq = d['sigmasq']
        beta = d['beta']
        load_num_grid = d['num_grid']
    assert num_grid == load_num_grid
    return dist, mu, x, y, sigmasq, beta, num_grid
