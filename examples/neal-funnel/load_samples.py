import os
import pickle


def load_samples():
    with open(os.path.join('data', 'samples.pkl'), 'rb') as f:
        d = pickle.load(f)
        iid = d['iid']
        num_dims = d['num_dims']
    return iid, num_dims
