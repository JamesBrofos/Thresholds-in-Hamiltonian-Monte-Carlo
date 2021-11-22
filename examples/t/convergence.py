import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
import tqdm

from load_samples import load_samples

def compute_ks(regex):
    fns = glob.glob(os.path.join('samples', '*' + regex + '*.pkl'))
    samples = np.zeros([len(fns), 10000])
    iid, Sigma, dof = load_samples(10000)

    with tqdm.tqdm(total=len(fns)) as pbar:
        for i, fn in enumerate(fns):
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                samples[i] = data['samples'][:, -1]
            pbar.update(1)

    num_steps = samples.shape[-1]
    ks = np.zeros(num_steps)
    with tqdm.tqdm(total=num_steps) as pbar:
        for i in range(num_steps):
            a = spst.kstest(samples[:, i], spst.t(df=5.0, loc=0.0, scale=np.sqrt(10000)).cdf).statistic
            ks[i] = a
            # b = spst.ks_2samp(samples[:, i], iid[:, -1]).statistic
            # print(a, b)
            pbar.update(1)

    return ks


euclid = compute_ks('num-samples-10000-method-euclidean*scale-10000')
rmn = {
    1e-1: compute_ks('thresh-0.1*num-samples-10000-method-riemannian*scale-10000'),
    1e-5: compute_ks('thresh-1e-05*num-samples-10000-method-riemannian*scale-10000'),
    1e-9: compute_ks('thresh-1e-09*num-samples-10000-method-riemannian*scale-10000'),
}

steps = np.arange(1, len(euclid) + 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(steps, euclid, label='Euclid. 0.1', alpha=0.8)
for k in rmn:
    ax.plot(steps, rmn[k], label='Thresh. {}'.format(k), alpha=0.8)

ax.set_xlabel('Sampling Iteration', fontsize=30)
ax.set_ylabel('KS Statistic', fontsize=30)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.legend(fontsize=30)
ax.grid(linestyle=':')
fig.tight_layout()
fig.savefig(os.path.join('images', 'convergence.pdf'))
