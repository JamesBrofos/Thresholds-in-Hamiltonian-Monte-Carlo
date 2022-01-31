import os
import pickle


def load_samples(fname):
    with open(os.path.join('data', fname), 'rb') as f:
        d = pickle.load(f)
        iid = d['iid']
        Sigma = d['Sigma']
        dof = d['dof']
    return iid, Sigma, dof
