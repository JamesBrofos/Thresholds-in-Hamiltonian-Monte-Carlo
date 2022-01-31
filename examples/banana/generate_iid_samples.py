import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from hmc.applications.banana import posterior_factory

import load_data

np.random.seed(0)


def rejection_sampler(log_posterior):
    logm = -50.5
    lp_max = -np.inf
    while True:
        x = np.random.uniform(low=(-30.0, -10.0), high=(10.0, 10.0))
        lp = log_posterior(x)
        logu = np.log(np.random.uniform())
        assert lp <= logm, lp
        if logu < lp - logm:
            yield x
        if lp > lp_max:
            lp_max = lp
            print(lp_max)

def main(num_samples=1000000, fname='samples.pkl'):
    y, sigma_y, sigma_theta = load_data.load_data()
    log_posterior, _, _, _, _ = posterior_factory(y, sigma_y, sigma_theta)

    sampler = rejection_sampler(log_posterior)
    samples = np.zeros([num_samples, 2])
    pbar = tqdm.tqdm(total=num_samples)
    for i in range(num_samples):
        x = next(sampler)
        samples[i] = x
        pbar.update(1)

    with open(os.path.join('data', fname), 'wb') as f:
        pickle.dump(samples, f)

if __name__ == '__main__':
    main()
    for i in range(10):
        main(1000000, 'samples-{}.pkl'.format(i+1))
