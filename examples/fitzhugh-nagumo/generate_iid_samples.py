import os
import pickle

import numpy as np
import tqdm

from hmc.applications import newton_raphson
from hmc.applications.fitzhugh_nagumo import posterior_factory
from hmc.linalg import solve_psd

from load_data import load_data


def rejection_sampler(mean, std, logm, log_posterior):
    k = 9
    low = mean - k*std
    high = mean + k*std
    while True:
        x = np.random.uniform(low=low, high=high)
        lp = log_posterior(x)
        logu = np.log(np.random.uniform())
        assert lp <= logm, lp
        if logu < lp - logm:
            yield x

def main(num_samples=100000, fname='samples.pkl'):
    y, t, sigma, state = load_data()
    (log_posterior, _, _, _, _, _, riemannian_auxiliaries) = posterior_factory(
        state, y, t, sigma)
    mean = newton_raphson(np.array([0.2, 0.2, 3.0]), riemannian_auxiliaries)
    lp, _, G, _ = riemannian_auxiliaries(mean)
    invG = solve_psd(G)
    logm = lp + 0.01
    std = np.sqrt(np.diag(invG))

    sampler = rejection_sampler(mean, std, logm, log_posterior)
    samples = np.zeros([num_samples, 3])
    pbar = tqdm.tqdm(total=num_samples)
    for i in range(num_samples):
        x = next(sampler)
        samples[i] = x
        pbar.update(1)

    with open(os.path.join('data', fname), 'wb') as f:
        pickle.dump(samples, f)

if __name__ == '__main__':
    # main()
    for i in range(10):
        main(100000, 'samples-{}.pkl'.format(i+1))
