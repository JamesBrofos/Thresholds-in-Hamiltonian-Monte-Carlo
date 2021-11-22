import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from hmc import metropolis_hastings, statistics, transforms
from hmc.applications.banana import posterior_factory
from hmc.proposals import RiemannianLeapfrogProposal

from load_data import load_data
from load_samples import load_samples


parser = argparse.ArgumentParser(description='Convergence effects in Riemannian HMC')
args = parser.parse_args()

def posterior_functions():
    y, sigma_y, sigma_theta = load_data()
    (
        log_posterior,
        metric,
        _,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    ) = posterior_factory(y, sigma_y, sigma_theta)
    return metric, riemannian_auxiliaries

metric, riemannian_auxiliaries = posterior_functions()

def main():
    id = '-'.join(
        '{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

    iid = load_samples()
    num_iid = len(iid)

    step_size = 0.04
    max_steps = 20
    max_iters = 1000

    base_proposal = RiemannianLeapfrogProposal(
        metric,
        riemannian_auxiliaries,
        1e-10,
        max_iters
    )

    num_trials = 1000
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)

    dist = np.zeros([len(thresholds), num_trials])
    rej = np.zeros_like(dist)

    for j in tqdm.tqdm(range(num_trials)):
        q = iid[np.random.choice(num_iid)]
        ns = int(np.ceil(max_steps*np.random.uniform()))
        unif = np.random.uniform()
        state = base_proposal.first_state(q)
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
