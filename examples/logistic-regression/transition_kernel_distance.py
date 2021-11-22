import argparse
import copy
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from hmc import metropolis_hastings, statistics, transforms
from hmc.applications.logistic import posterior_factory, load_dataset
from hmc.proposals import RiemannianLeapfrogProposal


parser = argparse.ArgumentParser(description='Convergence effects in Riemannian HMC')
args = parser.parse_args()

def posterior_functions(x, y, inv_alpha):
    (
        log_posterior,
        metric,
        _,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    ) = posterior_factory(x, y, inv_alpha)
    return metric, riemannian_auxiliaries

def main():
    id = '-'.join(
        '{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

    with open(os.path.join(
            'samples',
            'samples-thresh-0.0-max-iters-0-step-size-0.2-num-steps-20-num-samples-100000-method-euclidean-check-prob-0.01-dataset-heart-num-obs-270-newton-momentum-False-newton-position-False-seed-0.pkl'
    ), 'rb') as f:
        samples = pickle.load(f)['samples']

    num_samples = len(samples)
    x, y = load_dataset('heart')

    step_size = 0.2
    max_steps = 20
    max_iters = 100

    num_trials = 1000
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)

    dist = np.zeros([len(thresholds), num_trials])
    rej = np.zeros_like(dist)

    for j in tqdm.tqdm(range(num_trials)):
        q = samples[np.random.choice(num_samples)]
        b, inv_alpha = q[:-1], q[-1]

        metric, riemannian_auxiliaries = posterior_functions(x, y, inv_alpha)
        base_proposal = RiemannianLeapfrogProposal(
            metric,
            riemannian_auxiliaries,
            1e-10,
            max_iters
        )

        ns = int(np.ceil(max_steps*np.random.uniform()))
        unif = np.random.uniform()
        state = base_proposal.first_state(b)
        state.momentum = statistics.rvs(state.sqrtm_metric)
        _, _, q_base, acc_base = metropolis_hastings(
            base_proposal, copy.deepcopy(state), step_size, ns, unif,
            transforms.identity)
        for i, thresh in enumerate(thresholds):
            test_proposal = RiemannianLeapfrogProposal(
                metric,
                riemannian_auxiliaries,
                thresh,
                max_iters
            )
            _, _, q_test, acc_test = metropolis_hastings(
                test_proposal, copy.deepcopy(state), step_size, ns, unif,
                transforms.identity)
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
