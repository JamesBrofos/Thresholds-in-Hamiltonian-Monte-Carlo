import argparse
import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
import tqdm

from hmc import DualAveraging, RuppertAveraging, metropolis_hastings, statistics, transforms
from hmc.applications.neal_funnel import posterior_factory
from hmc.proposals import SoftAbsLeapfrogProposal

from load_samples import load_samples


alpha = 1e4
(
    log_posterior, hessian, euclidean_auxiliaries, riemannian_auxiliaries
) = posterior_factory()
base = SoftAbsLeapfrogProposal(alpha, hessian, riemannian_auxiliaries, 1e-10, 1000)
prop = SoftAbsLeapfrogProposal(alpha, hessian, riemannian_auxiliaries, 1e-10, 1000)

def loss(state, log_thresh, step_size, num_steps):
    prop.thresh = 10**log_thresh
    unif = np.random.uniform()
    base_state, _ = base.propose(copy.deepcopy(state), step_size, num_steps)
    q_base = base_state.position
    p_base = base_state.momentum
    prop_state, _ = prop.propose(copy.deepcopy(state), step_size, num_steps)
    q_test = prop_state.position
    p_test = prop_state.momentum
    new_state, _, _, _ = metropolis_hastings(
        prop,
        copy.deepcopy(state),
        step_size,
        num_steps,
        unif,
        transforms.identity
    )
    if prop.thresh < base.thresh:
        l = -10.0
    else:
        dist = np.sqrt(np.linalg.norm(q_base - q_test)**2 + np.linalg.norm(p_base - p_test)**2)
        dist = np.log10(np.maximum(dist, 1e-16))
        l = dist + 6.0
    return l, new_state


def main():
    step_size = 0.2
    num_steps = 25
    iid, num_dims = load_samples()
    avg = RuppertAveraging(-10.0, 0.75)
    loss_hist = []
    xb_hist = []
    x_hist = []
    qo = iid[np.random.choice(len(iid))]
    state = prop.first_state(qo)

    for i in range(1000):
        state.momentum = statistics.rvs(state.sqrtm_metric)
        H, state = loss(state, avg.x, step_size, num_steps)
        avg.update(H)
        print('iter. {} - loss: {:.3f} - xb: {:.3e}'.format(i+1, H, avg.xb))
        loss_hist.append(H)
        xb_hist.append(avg.xb)
        x_hist.append(avg.x)

    loss_hist = np.array(loss_hist)
    loss_summ = np.cumsum(loss_hist)
    loss_avg = loss_summ / (np.arange(len(loss_summ)) + 1.0)

    plt.figure()
    plt.plot(loss_hist - 6, label=r'$L_n$')
    plt.plot(loss_avg - 6, label=r'$\bar{L}_n$')
    plt.grid(linestyle=':')
    plt.gca().tick_params(axis='x', labelsize=24)
    plt.gca().tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=30)
    plt.xlabel('Iteration', fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'dual-averaging-loss.pdf'))

    plt.figure()
    plt.plot(x_hist, label=r'$\log_{10}\delta_n$')
    plt.plot(xb_hist, label=r'$\log_{10} \bar{\delta}_n$')
    plt.grid(linestyle=':')
    plt.gca().tick_params(axis='x', labelsize=24)
    plt.gca().tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=30)
    plt.xlabel('Iteration', fontsize=30)
    plt.ylim(-10.0, 0.0)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'dual-averaging-threshold.pdf'))

    thresholds = np.linspace(-10.0, -1, 100)
    num_trials = 500
    num_thresholds = len(thresholds)
    O = np.zeros([num_trials, num_thresholds])
    for i in tqdm.tqdm(range(num_trials)):
        qo = iid[np.random.choice(len(iid))]
        state = prop.first_state(qo)
        state.momentum = statistics.rvs(state.sqrtm_metric)
        ns = int(np.ceil(np.random.uniform()*num_steps))
        for j in range(num_thresholds):
            H, _ = loss(state, thresholds[j], step_size, ns)
            O[i, j] = H

    plt.figure()
    plt.plot(thresholds, O[:10].T - 6, 'k-', alpha=0.3)
    plt.plot(thresholds, O.mean(0) - 6, linewidth=3, label=r'Expected $L(\delta)$')
    plt.grid(linestyle=':')
    plt.xlabel('$\log_{10}$ Threshold', fontsize=30)
    plt.ylabel('$L(\delta)$', fontsize=30)
    plt.gca().tick_params(axis='x', labelsize=24)
    plt.gca().tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'dual-averaging-function.pdf'))


if __name__ == '__main__':
    main()
