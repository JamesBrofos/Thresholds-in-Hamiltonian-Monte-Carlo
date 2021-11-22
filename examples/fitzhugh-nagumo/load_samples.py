import os
import pickle


def load_samples():
    with open(os.path.join('data', 'samples.pkl'), 'rb') as f:
        iid = pickle.load(f)
    return iid
