import os
import pickle


def load_data():
    with open(os.path.join('data', 'data.pkl'), 'rb') as f:
        d = pickle.load(f)
        x = d['x']
        y = d['y']
        sigma = d['sigma']
        phi = d['phi']
        beta = d['beta']
    return x, y, sigma, phi, beta
