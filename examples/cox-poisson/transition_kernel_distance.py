import argparse
import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import hmc
from hmc.applications import cox_poisson as distr

from load_data import load_data


parser = argparse.ArgumentParser(description='Convergence effects in Riemannian HMC')
args = parser.parse_args()


def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

    with open(os.path.join(
            'samples',
            'samples-num-samples-5000-num-burn-1000-thresh-1e-10-max-iters-100-num-grid-32-method-riemannian-num-steps-hyper-5-step-size-hyper-0.2-num-steps-gaussian-50-step-size-gaussian-0.3-partial-momentum-0.0-check-prob-0.01-seed-0.pkl'
    ), 'rb') as f:
        samples = pickle.load(f)['samples']

    num_samples = len(samples)
    num_grid = 32
    cox_dist, mu, _, y, _, _, num_grid = load_data(num_grid)


    max_iters = 1000
    step_size = 0.2
    max_steps = 5
    num_trials = 1000

    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dist = np.zeros([num_thresholds, num_trials])
    rej = np.zeros_like(dist)

    for j in tqdm.tqdm(range(num_trials)):
        diff = np.zeros(num_trials)
        for i, thresh in enumerate(thresholds):
            s = samples[np.random.choice(num_samples)]
            x, q = s[:-2], s[-2:]
            (
                log_posterior,
                metric,
                _,
                euclidean_auxiliaries,
                riemannian_auxiliaries
            ) = distr.hyperparameter_posterior_factory(cox_dist, mu, x, y)

            base_proposal = hmc.proposals.RiemannianLeapfrogProposal(
                metric,
                riemannian_auxiliaries,
                1e-10,
                max_iters
            )
            test_proposal = hmc.proposals.RiemannianLeapfrogProposal(
                metric,
                riemannian_auxiliaries,
                thresh,
                max_iters
            )

            ns = int(np.ceil(max_steps*np.random.uniform()))
            unif = np.random.uniform()
            state = base_proposal.first_state(distr.forward_transform(q)[0])
            state.momentum = hmc.statistics.rvs(state.sqrtm_metric)
            _, _, q_base, acc_base = hmc.metropolis_hastings(
                base_proposal,
                copy.deepcopy(state),
                step_size,
                ns,
                unif,
                distr.inverse_transform
            )
            _, _, q_test, acc_test = hmc.metropolis_hastings(
                test_proposal,
                copy.deepcopy(state),
                step_size,
                ns,
                unif,
                distr.inverse_transform
            )
            dist[i, j] = np.linalg.norm(q_base - q_test)
            rej[i, j] = np.logical_and(not acc_base, not acc_test)

    log_dist = np.log10(dist)
    log_dist = np.where(np.isinf(log_dist), np.nan, log_dist)
    probs = rej.mean(axis=1)

    fig, ax1 = plt.subplots()
    bp = ax1.boxplot([_[~np.isnan(_)] for _ in log_dist], notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='tab:red')

    ax1.xaxis.set_tick_params(direction='out')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(np.arange(1, num_thresholds + 1))
    ax1.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=24)
    ax1.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax1.set_ylabel('$\log_{10}$ Proposal Distance', fontsize=30, color='tab:red')
    ax2 = ax1.twinx()
    ax2.bar(np.arange(1, num_thresholds + 1), probs, facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor='tab:blue')
    ax2.set_ylabel('Rejection Agreement', fontsize=30, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'transition-kernel-distance-{}.pdf'.format(id)))

if __name__ == '__main__':
    main()
