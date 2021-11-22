import argparse
import os
import pickle
import time

import numpy as np
import tqdm

from hmc import sample, summarize
from hmc.applications.neal_funnel import posterior_factory
from hmc.proposals import EuclideanLeapfrogProposal, SoftAbsLeapfrogProposal
from hmc.integrators.vectors import vector_field_from_softabs_auxiliaries, softabs_metric_handler


from load_samples import load_samples

parser = argparse.ArgumentParser(description='Convergence effects in Riemannian manifold HMC')
parser.add_argument('--thresh', type=float, default=0.0, help='Convergence threshold')
parser.add_argument('--max-iters', type=int, default=0, help='Maximum number of fixed point iterations')
parser.add_argument('--step-size', type=float, default=0.2, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=20, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to generate')
parser.add_argument('--method', type=str, default='euclidean', help='Specification of proposal operator method')
parser.add_argument('--alpha', type=float, default=1e4, help='SoftAbs sharpness parameter')
parser.add_argument('--partial-momentum', type=float, default=0.0, help='Momentum refreshment rate')
parser.add_argument('--check-prob', type=float, default=0.0, help='Probability of checking reversibility and volume preservation')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

np.random.seed(args.seed)
(
    log_posterior, hessian, euclidean_auxiliaries, riemannian_auxiliaries
) = posterior_factory()
vector_field = vector_field_from_softabs_auxiliaries(
    riemannian_auxiliaries, args.alpha)


def experiment(iid, step_size, num_steps, proposal, num_samples, partial_momentum, check_prob):
    idx = 985772
    qo = iid[idx]
    sampler = sample(
        qo,
        step_size,
        num_steps,
        proposal,
        partial_momentum=partial_momentum,
        check_prob=check_prob
    )
    samples = np.zeros([num_samples, len(qo)])
    pbar = tqdm.tqdm(total=num_samples, position=0, leave=True)
    elapsed = 0.0
    for i in range(num_samples):
        q, info, t = next(sampler)
        elapsed += t
        samples[i] = q
        pbar.set_postfix({
            'acc. prob.': info.accept.avg,
            'rev. err.': info.absrev.avg,
            'jac. det. err.': info.jacdet[1e-5].avg,
            'invalid': info.invalid.summ
        })
        pbar.update(1)
    metrics = summarize(samples)
    return samples, metrics, info, elapsed


def main():
    if args.method == 'softabs':
        assert args.thresh > 0.0 and args.max_iters > 0
    id = '-'.join(
        '{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')
    iid, num_dims = load_samples()

    Id = np.eye(num_dims + 1)
    proposal = {
        'euclidean': EuclideanLeapfrogProposal(euclidean_auxiliaries, Id),
        'softabs': SoftAbsLeapfrogProposal(
            args.alpha, hessian, riemannian_auxiliaries, args.thresh,
            args.max_iters),
    }[args.method]

    samples, metrics, info, elapsed = experiment(
        iid,
        args.step_size,
        args.num_steps,
        proposal,
        args.num_samples,
        args.partial_momentum,
        args.check_prob
    )

    with open(os.path.join('samples', 'samples-{}.pkl'.format(id)), 'wb') as f:
        dat = {
            'samples': samples,
            'relrev': np.array(info.relrev.values),
            'absrev': np.array(info.absrev.values),
            'jacdet': {k: np.array(v.values) for k, v in info.jacdet.items()},
            'time': elapsed,
        }
        if args.method == 'softabs':
            dat['nfp_pos'] = np.array(info.num_iters_pos.values)
            dat['nfp_mom'] = np.array(info.num_iters_mom.values)
        pickle.dump(dat, f)

if __name__ == '__main__':
    main()

