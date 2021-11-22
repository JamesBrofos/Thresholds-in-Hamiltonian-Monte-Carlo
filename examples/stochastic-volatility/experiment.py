import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from hmc import sample, summarize
from hmc.applications.stochastic_volatility import (
    volatility_posterior_factory,
    hyperparameter_posterior_factory,
    forward_transform, inverse_transform
)
from hmc.linalg import cholesky_tridiagonal, solve_tridiagonal
from hmc.proposals import EuclideanLeapfrogProposal, RiemannianLeapfrogProposal

from load_data import load_data


parser = argparse.ArgumentParser(description='Evaluation of threshold effects in the stochastic volatility model')
parser.add_argument('--num-samples', type=int, default=20000, help='Number of samples to generate')
parser.add_argument('--num-burn', type=int, default=10000, help='Number of burn-in samples')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence threshold')
parser.add_argument('--max-iters', type=int, default=10000, help='Maximum number of fixed point iterations')
parser.add_argument('--method', type=str, default='euclidean', help='Specification of proposal operator method')
parser.add_argument('--num-steps-hyper', type=int, default=50, help='Number of integration steps for the hyperparameters')
parser.add_argument('--partial-momentum', type=float, default=0.0, help='Momentum refreshment rate')
parser.add_argument('--newton-momentum', dest='newton_momentum', action='store_true', default=False, help='Enable Newton iterations for momentum fixed point solution')
parser.add_argument('--no-newton-momentum', dest='newton_momentum', action='store_false')
parser.add_argument('--newton-position', dest='newton_position', action='store_true', default=False, help='Enable Newton iterations for position fixed point solution')
parser.add_argument('--no-newton-position', dest='newton_position', action='store_false')
parser.add_argument('--check-prob', type=float, default=0.0, help='Probability of checking reversibility and volume preservation')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

np.random.seed(args.seed)


def sampler(x, q, y, num_burn, num_samples, method, thresh, max_iters, num_steps_hyper, partial_momentum, newton_momentum, newton_position, check_prob):
    if method in ('riemannian'):
        step_size_hyper = 0.5
        num_steps_volatility = 50
        step_size_volatility = 0.3
    elif method == 'euclidean':
        step_size_hyper = 0.01
        num_steps_volatility = 50
        step_size_volatility = 0.03

    Id_hyper = np.eye(3)
    Id_volatility = np.ones(len(x))

    num_total = num_burn + num_samples
    samples = np.zeros((num_total, len(y) + 3))
    pbar = tqdm.tqdm(total=num_total, position=0, leave=True)

    hyper_acc = 0.0
    hyper_absrev = []
    hyper_relrev = []
    hyper_jac = {}
    rev_avg = 0.0
    jac_avg = 0.0
    hyper_nfp_pos = []
    hyper_nfp_mom = []
    volatility_acc = 0.0

    num_rev = 0
    num_jac = 0
    elapsed = 0.0

    for i in range(num_total):
        # The sampling iteration.
        step = i + 1

        # Sample hyperparameters.
        (
            log_posterior,
            metric,
            log_posterior_and_metric,
            euclidean_auxiliaries,
            riemannian_auxiliaries
        ) = hyperparameter_posterior_factory(x, y)
        if method == 'riemannian':
            proposal_hyper = RiemannianLeapfrogProposal(
                metric,
                riemannian_auxiliaries,
                thresh,
                max_iters,
                newton_momentum,
                newton_position
            )
        elif method == 'euclidean':
            proposal_hyper = EuclideanLeapfrogProposal(
                euclidean_auxiliaries, Id_hyper)
        sampler = sample(
            q,
            step_size_hyper,
            num_steps_hyper,
            proposal_hyper,
            forward_transform,
            inverse_transform,
            partial_momentum=partial_momentum,
            check_prob=check_prob
        )
        q, info_hyper, t_hyper = next(sampler)

        # Sample volatilities.
        euclidean_auxiliaries, metric = volatility_posterior_factory(*q, y)
        if method in ('riemannian'):
            metric_volatility = metric()
            proposal_volatility = EuclideanLeapfrogProposal(
                euclidean_auxiliaries,
                metric_volatility,
                solve_tridiagonal,
                cholesky_tridiagonal
            )
        elif method == 'euclidean':
            proposal_volatility = EuclideanLeapfrogProposal(
                euclidean_auxiliaries, Id_volatility)
        sampler = sample(
            x,
            step_size_volatility,
            num_steps_volatility,
            proposal_volatility
        )
        x, info_volatility, t_volatility = next(sampler)
        elapsed += t_hyper + t_volatility

        samples[i] = np.hstack([x, q])
        if method == 'riemannian' and info_hyper.num_iters_pos.avg > 0:
            hyper_nfp_pos.append(info_hyper.num_iters_pos.avg)
        if method == 'riemannian' and info_hyper.num_iters_mom.avg > 0:
            hyper_nfp_mom.append(info_hyper.num_iters_mom.avg)
        if info_hyper.absrev.avg > -20:
            hyper_absrev.append(info_hyper.absrev.avg)
            hyper_relrev.append(info_hyper.relrev.avg)
            num_rev += 1
            w_rev = (num_rev - 1) / num_rev
            rev_avg = w_rev*rev_avg + info_hyper.absrev.avg / num_rev
        if info_hyper.jacdet[1e-5].avg > -20:
            for k in info_hyper.jacdet:
                if k in hyper_jac:
                    hyper_jac[k].append(info_hyper.jacdet[k].avg)
                else:
                    hyper_jac[k] = [info_hyper.jacdet[k].avg]
            num_jac += 1
            w_jac = (num_jac - 1) / num_jac
            jac_avg = w_jac*jac_avg + info_hyper.jacdet[1e-5].avg / num_jac
        w_acc = i / step
        hyper_acc = info_hyper.accept.avg / step + w_acc * hyper_acc
        volatility_acc = info_volatility.accept.avg / step + w_acc * volatility_acc

        d = {
            'hyp. acc. prob.': hyper_acc,
            'rev. err.': rev_avg,
            'jac. det. err.': jac_avg,
            'vol. acc. prob.': volatility_acc
        }
        if method == 'riemannian':
            d['num. mom.'] = info_hyper.num_iters_mom.avg
            d['num. pos.'] = info_hyper.num_iters_pos.avg
        pbar.set_postfix(d)
        pbar.update(1)

    # Exclude burn-in samples.
    samples = samples[num_burn:]
    out = (
        samples,
        hyper_acc,
        hyper_nfp_pos,
        hyper_nfp_mom,
        hyper_absrev,
        hyper_relrev,
        hyper_jac,
        volatility_acc,
        elapsed
    )
    return out

def main():
    if args.method == 'riemannian':
        assert args.thresh > 0.0 and args.max_iters > 0
    if args.method in ('euclidean'):
        assert args.thresh == 0.0 and args.max_iters == 0
    if args.method == 'euclidean':
         assert args.num_steps_hyper == 50

    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

    x, y, sigma, phi, beta = load_data()
    qinit = np.array([sigma, phi, beta])
    xinit = np.copy(x)

    (
        samples,
        hyper_acc,
        hyper_nfp_pos,
        hyper_nfp_mom,
        hyper_absrev,
        hyper_relrev,
        hyper_jac,
        volatility_acc,
        elapsed
    ) = sampler(
        xinit,
        qinit,
        y,
        args.num_burn,
        args.num_samples,
        args.method,
        args.thresh,
        args.max_iters,
        args.num_steps_hyper,
        args.partial_momentum,
        args.newton_momentum,
        args.newton_position,
        args.check_prob,
    )

    with open(os.path.join('samples', 'samples-{}.pkl'.format(id)), 'wb') as f:
        d = {
            'samples': samples,
            'absrev': np.array(hyper_absrev),
            'relrev': np.array(hyper_relrev),
            'jacdet': {k: np.array(v) for k, v in hyper_jac.items()},
            'time': elapsed
        }
        if args.method == 'riemannian':
            d['nfp_pos'] = np.array(hyper_nfp_pos)
            d['nfp_mom'] = np.array(hyper_nfp_mom)
        pickle.dump(d, f)

if __name__ == '__main__':
    main()
