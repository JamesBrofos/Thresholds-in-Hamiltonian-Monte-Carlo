import os
import pickle


def load_samples(scale):
    with open(os.path.join('data', 'samples-scale-{}.pkl'.format(scale)), 'rb') as f:
        d = pickle.load(f)
        iid = d['iid']
        Sigma = d['Sigma']
        dof = d['dof']
    return iid, Sigma, dof
