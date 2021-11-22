import os
import pickle


def load_data():
    with open(os.path.join('data', 'data.pkl'), 'rb') as f:
        d = pickle.load(f)
        y = d['y']
        t = d['t']
        sigma = d['sigma']
        state = d['state']
    return y, t, sigma, state
