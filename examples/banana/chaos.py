import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.stats as spst
import tqdm

from hmc.applications.banana import posterior_factory
from hmc.proposals import RiemannianLeapfrogProposal
from hmc.proposals.proposal import Diagnostics
from hmc.integrators.info import GeneralizedLeapfrogInfo
from hmc.integrators.stateful import generalized_leapfrog

from load_data import load_data


parser = argparse.ArgumentParser(description='Convergence effects in Riemannian manifold HMC')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence threshold')
parser.add_argument('--max-iters', type=int, default=1000, help='Maximum number of fixed point iterations')
parser.add_argument('--newton-momentum', dest='newton_momentum', action='store_true', default=False, help='Enable Newton iterations for momentum fixed point solution')
parser.add_argument('--no-newton-momentum', dest='newton_momentum', action='store_false')
parser.add_argument('--newton-position', dest='newton_position', action='store_true', default=False, help='Enable Newton iterations for position fixed point solution')
parser.add_argument('--no-newton-position', dest='newton_position', action='store_false')
args = parser.parse_args()

np.random.seed(0)

y, sigma_y, sigma_theta = load_data()
(
    log_posterior,
    metric,
    log_posterior_and_metric,
    euclidean_auxiliaries,
    riemannian_auxiliaries
) = posterior_factory(y, sigma_y, sigma_theta)
proposal = RiemannianLeapfrogProposal(
    metric,
    riemannian_auxiliaries,
    args.thresh,
    args.max_iters,
    args.newton_momentum,
    args.newton_position
)
qo = np.array([ 0.7260814,  -0.51240169])
po = np.array([ 6.74542356, -6.2053273 ])
if False:
    qo *= 0.1
    po *= 0.1


state = proposal.first_state(qo)
state.momentum = po
info = GeneralizedLeapfrogInfo()
G = metric(state.position)
Mp = np.random.multivariate_normal(np.zeros(2), G, 10000)

def simulate(state, step_size, num_steps, u):
    fwd_state, fwd_info = generalized_leapfrog(
        state,
        step_size,
        num_steps,
        metric,
        riemannian_auxiliaries,
        args.thresh,
        args.max_iters,
        args.newton_momentum,
        args.newton_position
    )
    diff = np.zeros_like(wr)
    for i, w in enumerate(wr):
        per_state = proposal.first_state(state.position)
        per_state.momentum = state.momentum + w*u
        new_state, new_info = generalized_leapfrog(
            per_state,
            step_size,
            num_steps,
            metric,
            riemannian_auxiliaries,
            args.thresh,
            args.max_iters,
            args.newton_momentum,
            args.newton_position
        )
        diff[i] = np.sqrt(
            np.linalg.norm(fwd_state.position - new_state.position)**2 +
            np.linalg.norm(fwd_state.momentum - new_state.momentum)**2)
    return diff

u = np.random.normal(size=(2, ))
u /= np.linalg.norm(u)

wr = np.logspace(-13, -1, 13)

plt.figure()
diff = simulate(state, 0.04, 25, u)
plt.loglog(wr, diff, '-', label=r'$\epsilon = 0.04, k = 25$')
diff = simulate(state, 0.01, 100, u)
plt.loglog(wr, diff, '-', label=r'$\epsilon = 0.01, k = 100$')
plt.legend(fontsize=16)
plt.grid(linestyle=':')
plt.xlabel('Perturbation Size', fontsize=25)
plt.ylabel('Sensitivity', fontsize=25)
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='x', labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join('images', 'chaos.pdf'))

exit()

def grid(step_size, num_steps, newton_momentum, newton_position):
    proposal = RiemannianLeapfrogProposal(
        metric,
        riemannian_auxiliaries,
        args.thresh,
        args.max_iters,
        newton_momentum,
        newton_position
    )
    proposal.info.jacdet = {1e-5: Diagnostics()}
    lin = np.linspace(-30.0, 30.0, 10)
    px, py = np.meshgrid(lin, lin)
    pxr = np.ravel(px)
    pyr = np.ravel(py)
    M = np.array([pxr, pyr]).T

    for i in tqdm.tqdm(range(len(M))):
        state = proposal.first_state(qo)
        state.momentum = M[i]
        proposal.random_check(state, step_size, num_steps, info)

    D = np.array(proposal.info.jacdet[1e-5].values).reshape(px.shape)
    return px, py, D

levels = np.linspace(-9, 0, 10)
L = np.linalg.cholesky(G)
Linv = spla.solve_triangular(L, np.eye(2), lower=True)

r = np.sqrt(spst.chi2.ppf(0.99, 2))
t = np.linspace(0.0, 2*np.pi, 1000)
c = np.array([np.cos(t), np.sin(t)]).T
el = (np.linalg.cholesky(G)@c.T).T

px, py, D = grid(0.04, 25, False, False)
plt.figure()
plt.contourf(px, py, D, cmap=plt.cm.jet, levels=levels)
plt.plot(el[:, 0], el[:, 1], 'k-')
plt.xlabel('$p_1$', fontsize=25)
plt.ylabel('$p_2$', fontsize=25)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join('images', 'chaos-determinant-fixed-point-unstable.pdf'))

px, py, D = grid(0.04, 25, True, True)
plt.figure()
plt.contourf(px, py, D, cmap=plt.cm.jet, levels=levels)
plt.plot(el[:, 0], el[:, 1], 'k-')
plt.xlabel('$p_1$', fontsize=25)
plt.ylabel('$p_2$', fontsize=25)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join('images', 'chaos-determinant-newton-unstable.pdf'))

px, py, D = grid(0.01, 100, True, True)
plt.figure()
plt.contourf(px, py, D, cmap=plt.cm.jet, levels=levels)
plt.plot(el[:, 0], el[:, 1], 'k-')
plt.xlabel('$p_1$', fontsize=25)
plt.ylabel('$p_2$', fontsize=25)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join('images', 'chaos-determinant-newton-stable.pdf'))

