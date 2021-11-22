import os

import matplotlib.pyplot as plt
import numpy as np

import hmc
from hmc.integrators.stateful import leapfrog, generalized_leapfrog, lagrangian_leapfrog
from hmc.integrators import states
from hmc.applications import banana

import load_data
import load_samples



y, sigma_y, sigma_theta = load_data.load_data()
(
    log_posterior,
    metric,
    _,
    euclidean_auxiliaries,
    riemannian_auxiliaries
) = banana.posterior_factory(y, sigma_y, sigma_theta)
del y, sigma_y, sigma_theta

iid = load_samples.load_samples()
num_iid = len(iid)


def posterior_grid(num_lin):
    t1, t2 = np.meshgrid(np.linspace(-4.0, 2.0, num_lin),
                         np.linspace(-2.5, 2.5, num_lin))
    grid = np.stack((np.ravel(t1), np.ravel(t2)), axis=-1)
    num_grid = len(grid)
    lp = np.zeros(num_grid)
    for i in range(num_grid):
        lp[i] = log_posterior(grid[i])

    lp = lp - lp.max()
    lp = np.reshape(lp, t1.shape)
    prob = np.exp(lp)
    return t1, t2, prob

def riemannian_integrator(q, p, num_steps, step_size, thresh):
    G = metric(q)
    L = np.linalg.cholesky(G)
    p = L@p
    state = states.RiemannianLeapfrogState(q, p)
    state.logdet_metric = 2*np.sum(np.log(np.diag(L)))
    state.update(riemannian_auxiliaries)
    traj = np.zeros([num_steps, 2])
    traj[0] = state.position
    ham = np.zeros(num_steps)
    ham[0] = hmc.hamiltonian.hamiltonian(
        state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    for i in range(num_steps):
        state, info = generalized_leapfrog(
            state,
            step_size,
            1,
            metric,
            riemannian_auxiliaries,
            thresh,
            100,
            False
        )
        traj[i] = state.position
        ham[i] = hmc.hamiltonian.hamiltonian(
            state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    ham = np.abs(ham - ham[0])
    return traj, ham

def lagrangian_integrator(q, p, num_steps, step_size):
    G = metric(q)
    L = np.linalg.cholesky(G)
    p = L@p
    state = states.LagrangianLeapfrogState(q, p)
    state.logdet_metric = 2*np.sum(np.log(np.diag(L)))
    state.update(riemannian_auxiliaries)
    traj = np.zeros([num_steps, 2])
    traj[0] = state.position
    ham = np.zeros(num_steps)
    ham[0] = hmc.hamiltonian.hamiltonian(
        state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    for i in range(num_steps):
        state, info = lagrangian_leapfrog(
            state,
            step_size,
            1,
            riemannian_auxiliaries,
        )
        traj[i] = state.position
        ham[i] = hmc.hamiltonian.hamiltonian(
            state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    ham = np.abs(ham - ham[0])
    return traj, ham

def euclidean_integrator(q, p, num_steps, step_size):
    Id = np.eye(2)
    state = states.EuclideanLeapfrogState(q, p)
    state.inv_metric = Id
    state.logdet_metric = 0.0
    state.update(euclidean_auxiliaries)
    traj = np.zeros([num_steps, 2])
    traj[0] = state.position
    ham = np.zeros(num_steps)
    ham[0] = hmc.hamiltonian.hamiltonian(
        state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    for i in range(num_steps):
        state, info = leapfrog(
            state,
            step_size,
            1,
            euclidean_auxiliaries
        )
        traj[i] = state.position
        ham[i] = hmc.hamiltonian.hamiltonian(
            state.momentum, state.log_posterior, state.logdet_metric, state.inv_metric)
    ham = np.abs(ham - ham[0])
    return traj, ham

def main():
    np.random.seed(1)

    num_lin = 100
    t1, t2, prob = posterior_grid(num_lin)

    num_steps = 200
    step_size = 0.04

    q = iid[np.random.choice(num_iid)]
    p = np.random.normal(size=q.shape)
    etraj, eham = euclidean_integrator(q, p, num_steps, step_size)
    rbase, _ = riemannian_integrator(q, p, num_steps, step_size, 1e-10)
    # ltraj, _ = lagrangian_integrator(q, p, num_steps, step_size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(t1, t2, prob, cmap=plt.cm.Greys)
    for thresh in [1e-1, 1e-3, 1e-9]:
        rtraj, rham = riemannian_integrator(q, p, num_steps, step_size, thresh)
        ax.plot(rtraj[:, 0], rtraj[:, 1], label='{:.0e}'.format(thresh), zorder=3)

    ax.plot(etraj[:, 0], etraj[:, 1], label='Euclidean', alpha=1.0, zorder=2)
    # ax.plot(ltraj[:, 0], ltraj[:, 1], label='Lagrangian', alpha=1.0, zorder=2)
    ax.legend(fontsize=24)
    ax.grid(linestyle=':')
    ax.set_xlabel(r'$\theta_1$', fontsize=30)
    ax.set_ylabel(r'$\theta_2$', fontsize=30)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'threshold-precondition.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for thresh in [1e-1, 1e-3, 1e-9]:
        rtraj, _ = riemannian_integrator(q, p, num_steps, step_size, thresh)
        e = np.linalg.norm(rtraj - rbase, axis=-1)
        e = np.log10(np.maximum(e, 1e-16))
        ax.plot(np.arange(len(e)) + 1, e, label='{:.0e}'.format(thresh))
    ax.grid(linestyle=':')
    ax.set_xlabel('Integration Step', fontsize=30)
    ax.set_ylabel(r'$\log_{10}$ Position Error', fontsize=30)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'position-deviation.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for thresh in [1e-1, 1e-5, 1e-9]:
        rtraj, rham = riemannian_integrator(q, p, num_steps, step_size, thresh)
        rham = np.log10(np.maximum(rham, 1e-16))
        ax.plot(np.arange(len(rham)) + 1, rham)

    ax.grid(linestyle=':')
    ax.set_xlabel('Integration Step', fontsize=30)
    ax.set_ylabel(r'$\log_{10}$ Energy Differential', fontsize=30)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'energy-deviation.pdf'))

if __name__ == '__main__':
    main()
