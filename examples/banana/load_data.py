import os
import pickle


def load_data():
    with open(os.path.join('data', 'data.pkl'), 'rb') as f:
        d = pickle.load(f)
        y = d['y']
        sigma_y = d['sigma_y']
        sigma_theta = d['sigma_theta']
    return y, sigma_y, sigma_theta
