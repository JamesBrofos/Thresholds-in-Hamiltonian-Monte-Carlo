import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
import tqdm

import hmc
from hmc.applications import logistic, newton_raphson


parser = argparse.ArgumentParser(description='Convergence effects in Riemannian HMC')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence threshold')
parser.add_argument('--max-iters', type=int, default=10000, help='Maximum number of fixed point iterations')
parser.add_argument('--step-size', type=float, default=0.1, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=20, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to generate')
parser.add_argument('--method', type=str, default='euclidean', help='Specification of proposal operator method')
parser.add_argument('--check-prob', type=float, default=0.0, help='Probability of checking reversibility and volume preservation')
parser.add_argument('--dataset', type=str, default='heart', help='Specification of which logistic regression dataset to use')
parser.add_argument('--num-obs', type=int, default=270, help='Number of observations to use in logistic regression likelihood')
parser.add_argument('--newton-momentum', dest='newton_momentum', action='store_true', default=False, help='Enable Newton iterations for momentum fixed point solution')
parser.add_argument('--no-newton-momentum', dest='newton_momentum', action='store_false')
parser.add_argument('--newton-position', dest='newton_position', action='store_true', default=False, help='Enable Newton iterations for position fixed point solution')
parser.add_argument('--no-newton-position', dest='newton_position', action='store_false')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def sampler(x, y, step_size, num_steps, num_samples, method, thresh, max_iters, newton_momentum, newton_position, check_prob):
    num_burn = 1000
    num_total = num_samples + num_burn
    num_coef = x.shape[-1]
    Id = np.eye(num_coef)
    samples = np.zeros([num_total, num_coef + 1])
    b = np.zeros(num_coef)

    avgacc = 0
    acc = []
    absrev = []
    relrev = []
    jac = {}
    nfp_pos = []
    nfp_mom = []
    nfp_mom_avg = 0.0
    nfp_pos_avg = 0.0
    rev_avg = 0.0
    jac_avg = 0.0
    num_rev = 0
    num_jac = 0

    elapsed = 0.0
    pbar = tqdm.tqdm(total=num_total, position=0, leave=True)
    for i in range(num_total):
        inv_alpha = logistic.sample_posterior_precision(b, 1.0, 2.0)
        (
            log_posterior,
            metric,
            log_posterior_and_metric,
            euclidean_auxiliaries,
            riemannian_auxiliaries
        ) = logistic.posterior_factory(x, y, inv_alpha)
        if method == 'euclidean':
            proposal = hmc.proposals.EuclideanLeapfrogProposal(
                euclidean_auxiliaries,
                Id
            )
        elif method == 'riemannian':
            proposal = hmc.proposals.RiemannianLeapfrogProposal(
                metric,
                riemannian_auxiliaries,
                thresh,
                max_iters,
                newton_momentum,
                newton_position
            )

        if i == 0:
            b = newton_raphson(b, riemannian_auxiliaries)

        sampler = hmc.sample(b, step_size, num_steps, proposal, check_prob=check_prob)
        b, info, t = next(sampler)
        elapsed += t
        samples[i] = np.hstack([b, inv_alpha])

        step = i + 1
        w = (step - 1) / step
        if method == 'riemannian':
            if info.num_iters_pos.avg > 0:
                v = info.num_iters_pos.avg
                nfp_pos.append(v)
                nfp_pos_avg = w*nfp_pos_avg + v/step
            if info.num_iters_mom.avg > 0:
                v = info.num_iters_mom.avg
                nfp_mom.append(v)
                nfp_mom_avg = w*nfp_mom_avg + v/step
        if info.absrev.avg > -20:
            absrev.append(info.absrev.avg)
            relrev.append(info.relrev.avg)
            num_rev += 1
            w_rev = (num_rev - 1) / num_rev
            rev_avg = w_rev*rev_avg + info.absrev.avg / num_rev
        if info.jacdet[1e-5].avg > -20:
            for k in info.jacdet:
                if k in jac:
                    jac[k].append(info.jacdet[k].avg)
                else:
                    jac[k] = [info.jacdet[k].avg]
            num_jac += 1
            w_jac = (num_jac - 1) / num_jac
            jac_avg = w_jac*jac_avg + info.jacdet[1e-5].avg / num_jac

        acc.append(info.accept.avg)
        avgacc = w*avgacc + (1-w)*info.accept.avg
        d = {
            'acc. prob.': avgacc,
            'rev. err.': rev_avg,
            'jac. det. err.': jac_avg,
        }
        if method == 'riemannian':
            d['num. mom.'] = nfp_mom_avg
            d['num. pos.'] = nfp_pos_avg

        pbar.set_postfix(d)
        pbar.update(1)
    samples = samples[num_burn:]
    return samples, acc, nfp_pos, nfp_mom, absrev, relrev, jac, elapsed

def main():
    if args.method in ('euclidean'):
        assert args.thresh == 0.0 and args.max_iters == 0
    if args.method != 'riemannian':
        assert not args.newton_momentum and not args.newton_position

    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')
    print(id)

    x, y = logistic.data_loader.load_dataset(args.dataset)
    x, y = x[:args.num_obs], y[:args.num_obs]
    print('num. obs.: {} - num. feats.: {}'.format(*x.shape))

    np.random.seed(args.seed)
    samples, acc, nfp_pos, nfp_mom, absrev, relrev, jac, elapsed = sampler(
        x,
        y,
        args.step_size,
        args.num_steps,
        args.num_samples,
        args.method,
        args.thresh,
        args.max_iters,
        args.newton_momentum,
        args.newton_position,
        args.check_prob,
    )
    print('acceptance prob.: {:.4f}'.format(np.mean(acc)))
    metrics = hmc.summarize(samples)

    with open(os.path.join('samples', 'samples-{}.pkl'.format(id)), 'wb') as f:
        dat = {
            'samples': samples,
            'absrev': np.array(absrev),
            'relrev': np.array(relrev),
            'jacdet': {k: np.array(v) for k, v in jac.items()},
            'time': elapsed
        }
        if args.method == 'riemannian':
            dat['nfp_pos'] = np.array(nfp_pos)
            dat['nfp_mom'] = np.array(nfp_mom)
        pickle.dump(dat, f)


if __name__ == '__main__':
    main()
