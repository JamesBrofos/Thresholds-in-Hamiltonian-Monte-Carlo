import argparse
import os
import pickle

import numpy as np
import scipy.stats as spst
from scipy.spatial.distance import cdist

from hmc import mmd

from load_samples import load_samples
from unit_vectors import load_unit_vectors


parser = argparse.ArgumentParser(description='Compute ergodicity metrics for the Neal funnel distribution')
parser.add_argument('--file-name', type=str, default='', help='Identifier of which samples file to parse')
args = parser.parse_args()

def compute_ks(iid, samples):
    U = load_unit_vectors()
    ks = np.zeros(len(U))
    for i, u in enumerate(U):
        ks[i] = spst.ks_2samp(samples@u, iid@u).statistic
    return ks

def compute_mmd(iid, samples):
    bw = 8.4
    u = mmd(samples, iid, bw)
    return u

def compute_sliced_wasserstein(iid, samples):
    U = load_unit_vectors()
    W = np.zeros(len(U))
    for i, u in enumerate(U):
        x = samples@u
        y = iid@u
        W[i] = spst.wasserstein_distance(x, y)
    sw = np.mean(W)
    return sw

def main():
    iid, num_dims = load_samples()
    with open(args.file_name, 'rb') as f:
        data = pickle.load(f)
    samples = data['samples']
    if 'ks' not in data:
        data['ks'] = compute_ks(iid, samples)
        with open(args.file_name, 'wb') as f:
            pickle.dump(data, f)
    if 'mmd' not in data:
        data['mmd'] = compute_mmd(iid, samples)
        with open(args.file_name, 'wb') as f:
            pickle.dump(data, f)
    if 'sw' not in data:
        data['sw'] = compute_sliced_wasserstein(iid, samples)
        with open(args.file_name, 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    main()
