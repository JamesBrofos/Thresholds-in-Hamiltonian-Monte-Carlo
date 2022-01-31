import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from hmc import summarize

from load_data import load_data

def euclidean_samples():
    ng = 32
    num_samples = [100000] if ng == 16 else [5000]
    euclid = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*num-samples-{}*euclidean*'.format(ns))))
        for f in fns:
            ss = 0.01
            with open(f, 'rb') as g:
                d[ss] = pickle.load(g)
        euclid[ns] = d
    return euclid

def riemannian_samples(newton_momentum=False, newton_position=False):
    ng = 32
    num_samples = [100000] if ng == 16 else [5000]
    rmn = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*num-samples-{}*riemannian*-newton-momentum-{}*newton-position-{}*'.format(ns, newton_momentum, newton_position))))
        for f in fns:
            t = f.split('-thresh-')[1].split('-m')[0]
            t = float(t)
            with open(f, 'rb') as g:
                d[t] = pickle.load(g)
        rmn[ns] = d
    return rmn

def latent_gaussian():
    num_grid = 32
    dist, mu, x, y, sigmasq, beta, num_grid = load_data(num_grid)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.imshow(x.reshape(num_grid, num_grid), cmap=plt.cm.turbo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(122)
    ax.imshow(y.reshape(num_grid, num_grid), cmap=plt.cm.turbo)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'latent-gaussian-process.png'))

def cox_poisson():
    euclid = euclidean_samples()
    euclid = euclid[max(euclid.keys())]
    rmn = riemannian_samples()
    rmn = rmn[max(rmn.keys())]

    num_grid = int(np.sqrt(euclid[0.01]['samples'][0][:-2].size))

    k = len(rmn) + len(euclid)
    fig = plt.figure(figsize=(30, 8))
    for i, t in tqdm.tqdm(enumerate(euclid.keys())):
        g = euclid[t]['samples'][:, :-2]
        g = g.reshape((-1, num_grid, num_grid))
        ax = fig.add_subplot(2, k, i+1)
        ax.imshow(g.mean(0), cmap=plt.cm.turbo)
        ax.set_title('Euclid. {:.0e}'.format(t), fontsize=35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(2, k, i+k+1)
        ax.imshow(g.std(0), cmap=plt.cm.turbo)
        ax.set_xticks([])
        ax.set_yticks([])

    rkeys = rmn.keys()
    rkeys = sorted(rkeys, reverse=False)
    for i, t in tqdm.tqdm(enumerate(rkeys)):
        g = rmn[t]['samples'][:, :-2]
        g = g.reshape((-1, num_grid, num_grid))
        ax = fig.add_subplot(2, k, i+1+len(euclid))
        ax.imshow(g.mean(0), cmap=plt.cm.turbo)
        ax.set_title('Thresh. {:.0e}'.format(t), fontsize=35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(2, k, i+k+1+len(euclid))
        ax.imshow(g.std(0), cmap=plt.cm.turbo)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'cox-poisson.png'))

def gaussian_effective_sample_size():
    euclid = euclidean_samples()
    euclid = euclid[max(euclid.keys())]
    rmn = riemannian_samples()
    rmn = rmn[max(rmn.keys())]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    labels = ['Euclidean {}'.format(t) for t in ekeys] + ['Threshold {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    ess = {}
    for t in tqdm.tqdm(ekeys):
        breaks = np.split(euclid[t]['samples'][:, :-2], 10, axis=0)
        k = 'euclid-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].mean()
            ess[k].append(m)

    for t in tqdm.tqdm(rkeys):
        breaks = np.split(rmn[t]['samples'][:, :-2], 10, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].mean()
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True)
    for i, k in enumerate(ess.keys()):
        ax.plot((i+1)*np.ones(len(ess[k])), ess[k], '.', color='tab:orange')

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Minimum ESS')
    ax.grid(linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'volatility-minimum-ess.pdf'))

def hyper_parameter_effective_sample_size():
    euclid = euclidean_samples()
    euclid = euclid[max(euclid.keys())]
    rmn = riemannian_samples()
    rmn = rmn[max(rmn.keys())]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)

    labels = ['Euclidean {}'.format(t) for t in ekeys] + ['Threshold {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    num_breaks = 10
    ess = {}
    for t in ekeys:
        breaks = np.split(euclid[t]['samples'][:, -2:], num_breaks, axis=0)
        k = 'euclid-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].mean()
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        breaks = np.split(rmn[t]['samples'][:, -2:], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].mean()
            ess[k].append(m)

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Minimum ESS')
    ax.grid(linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'hyper-parameter-minimum-ess.pdf'))

def hyper_parameter_effective_sample_size_per_second():
    euclid = euclidean_samples()
    euclid = euclid[max(euclid.keys())]
    rmn = riemannian_samples()
    rmn = rmn[max(rmn.keys())]
    nm_rmn = riemannian_samples(True)
    nm_rmn = nm_rmn[max(nm_rmn.keys())]
    nb_rmn = riemannian_samples(True, True)
    nb_rmn = nb_rmn[max(nb_rmn.keys())]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)

    for vidx in range(1, 3):
        labels = ['Euclid. {}'.format(t) for t in ekeys] + ['Thresh. {:.0e}'.format(t) for t in rkeys]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        num_breaks = 10
        ess = {}
        for t in ekeys:
            breaks = np.split(euclid[t]['samples'][:, [-vidx]], num_breaks, axis=0)
            k = 'euclid-{}'.format(t)
            ess[k] = []
            for i, b in enumerate(breaks):
                metrics = summarize(b)
                m = metrics['ess'].mean() / (euclid[t]['time'] / num_breaks)
                ess[k].append(m)

        ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True, showextrema=False)

        ess = {}
        for t in rkeys:
            breaks = np.split(rmn[t]['samples'][:, [-vidx]], num_breaks, axis=0)
            k = 'rmn-{}'.format(t)
            ess[k] = []
            for i, b in enumerate(breaks):
                metrics = summarize(b)
                m = metrics['ess'].mean() / (rmn[t]['time'] / num_breaks)
                ess[k].append(m)

        vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(['' for l in labels])
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
        ax.set_xlabel('')
        ax.set_ylabel('ESS / Sec.', fontsize=18)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)
        ax.grid(linestyle=':')

        fig.tight_layout()
        fig.savefig(os.path.join('images', 'hyper-parameter-minimum-ess-per-second-{}.pdf'.format(vidx)))

        labels = ['Thresh. {:.0e}'.format(t) for t in rkeys]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        num_breaks = 10

        ess = {}
        for t in rkeys:
            breaks = np.split(rmn[t]['samples'][:, [-vidx]], num_breaks, axis=0)
            k = 'rmn-{}'.format(t)
            ess[k] = []
            for i, b in enumerate(breaks):
                metrics = summarize(b)
                m = metrics['ess'].mean() / (rmn[t]['time'] / num_breaks)
                ess[k].append(m)

        vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

        ess = {}
        for t in rkeys:
            breaks = np.split(nm_rmn[t]['samples'][:, [-vidx]], num_breaks, axis=0)
            k = 'rmn-{}'.format(t)
            ess[k] = []
            for i, b in enumerate(breaks):
                metrics = summarize(b)
                m = metrics['ess'].mean() / (nm_rmn[t]['time'] / num_breaks)
                ess[k].append(m)

        vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

        ess = {}
        for t in rkeys:
            breaks = np.split(nb_rmn[t]['samples'][:, [-vidx]], num_breaks, axis=0)
            k = 'rmn-{}'.format(t)
            ess[k] = []
            for i, b in enumerate(breaks):
                metrics = summarize(b)
                m = metrics['ess'].mean() / (nb_rmn[t]['time'] / num_breaks)
                ess[k].append(m)

        vpd = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)


        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(['' for l in labels])
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel('')
        ax.set_ylabel('ESS / Sec.', fontsize=18)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)
        ax.grid(linestyle=':')
        if vidx == 1:
            ax.legend([vpb["bodies"][0], vpc["bodies"][0], vpd["bodies"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=16, loc='upper left')

        fig.tight_layout()
        fig.savefig(os.path.join('images', 'hyper-parameter-minimum-ess-per-second-vs-newton-{}.pdf'.format(vidx)))


def volume_preservation():
    rmn = riemannian_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    mkey = max(rmn.keys())
    dat = [rmn[mkey][t]['jacdet'][1e-3] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Vol. Pres. Err.', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'jacobian-determinant.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='tab:blue')

    nm_rmn = riemannian_samples(True)
    dat = [nm_rmn[5000][t]['jacdet'][1e-3] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nm_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nm_bp['boxes']:
        patch.set(facecolor='tab:red')

    nb_rmn = riemannian_samples(True, True)
    dat = [nb_rmn[5000][t]['jacdet'][1e-3] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nb_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nb_bp['boxes']:
        patch.set(facecolor='tab:green')

    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Vol. Pres. Err.', fontsize=30)
    # ax.legend([bp["boxes"][0], nm_bp["boxes"][0], nb_bp["boxes"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=20, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'jacobian-determinant-vs-newton.pdf'))

    perturb = sorted(rmn[mkey][1e-9]['jacdet'].keys())
    num_perturb = len(perturb)
    dat = [rmn[mkey][1e-9]['jacdet'][p] for p in perturb]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_perturb + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in perturb], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, num_perturb + 0.75)
    ax.set_xlabel('$\log_{10}$ Perturbation', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Volume Preservation Error', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'perturbation.pdf'))

def reversibility():
    rmn = riemannian_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [rmn[max(rmn.keys())][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Abs. Rev. Error', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'absolute-reversibility.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='tab:blue')

    nm_rmn = riemannian_samples(True)
    dat = [nm_rmn[5000][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nm_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nm_bp['boxes']:
        patch.set(facecolor='tab:red')

    nb_rmn = riemannian_samples(True, True)
    dat = [nb_rmn[5000][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nb_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nb_bp['boxes']:
        patch.set(facecolor='tab:green')

    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Abs. Rev. Err.', fontsize=30)
    # ax.legend([bp["boxes"][0], nm_bp["boxes"][0], nb_bp["boxes"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=20, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'absolute-reversibility-vs-newton.pdf'))

    dat = [rmn[max(rmn.keys())][t]['relrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Rel. Rev. Error', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'relative-reversibility.pdf'))

def momentum_fixed_point():
    rmn = riemannian_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [np.log10(rmn[max(rmn.keys())][t]['nfp_mom']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Momentum Fixed Point', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-momentum.pdf'))

    nrmn = riemannian_samples(True)
    dat = [np.log10(rmn[max(rmn.keys())][t]['nfp_mom']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    mean = np.array([np.mean(_) for _ in dat])
    std = np.array([np.std(_) for _ in dat])
    ndat = [np.log10(nrmn[max(rmn.keys())][t]['nfp_mom']) for t in thresholds]
    ndat = [_[~np.isnan(_)] for _ in ndat]
    nmean = np.array([np.mean(_) for _ in ndat])
    nstd = np.array([np.std(_) for _ in ndat])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, num_thresholds + 1), mean, color='tab:blue', label='Fixed Point')
    ax.plot(np.arange(1, num_thresholds + 1), mean + std, '--', color='tab:blue')
    ax.plot(np.arange(1, num_thresholds + 1), mean - std, '--', color='tab:blue')
    ax.plot(np.arange(1, num_thresholds + 1), nmean, color='tab:orange', label='Newton')
    ax.plot(np.arange(1, num_thresholds + 1), nmean + nstd, '--', color='tab:orange')
    ax.plot(np.arange(1, num_thresholds + 1), nmean - nstd, '--', color='tab:orange')
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Momentum Fixed Point', fontsize=20)
    ax.set_ylim((0.0, 1.1))
    ax.legend(fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-momentum-vs-newton.pdf'))

def position_fixed_point():
    rmn = riemannian_samples()
    nrmn = riemannian_samples(True, True)
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [np.log10(rmn[max(rmn.keys())][t]['nfp_pos']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dat, notch=True)
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Position Fixed Point', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-position.pdf'))

    dat = [np.log10(rmn[max(rmn.keys())][t]['nfp_pos']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    mean = np.array([np.mean(_) for _ in dat])
    std = np.array([np.std(_) for _ in dat])
    ndat = [np.log10(nrmn[max(nrmn.keys())][t]['nfp_pos']) for t in thresholds]
    ndat = [_[~np.isnan(_)] for _ in ndat]
    nmean = np.array([np.mean(_) for _ in ndat])
    nstd = np.array([np.std(_) for _ in ndat])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, num_thresholds + 1), mean, color='tab:blue', label='Fixed Point')
    ax.plot(np.arange(1, num_thresholds + 1), mean + std, '--', color='tab:blue')
    ax.plot(np.arange(1, num_thresholds + 1), mean - std, '--', color='tab:blue')
    ax.plot(np.arange(1, num_thresholds + 1), nmean, color='tab:orange', label='Newton')
    ax.plot(np.arange(1, num_thresholds + 1), nmean + nstd, '--', color='tab:orange')
    ax.plot(np.arange(1, num_thresholds + 1), nmean - nstd, '--', color='tab:orange')
    ax.grid(linestyle=':')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, num_thresholds + 1))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlim(0.25, len(thresholds) + 0.75)
    ax.set_xlabel('$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel('$\log_{10}$ Position Fixed Point', fontsize=20)
    ax.set_ylim((0.0, 1.1))
    ax.legend(fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-position-vs-newton.pdf'))


def main():
    momentum_fixed_point()
    position_fixed_point()

    latent_gaussian()
    cox_poisson()
    hyper_parameter_effective_sample_size()
    hyper_parameter_effective_sample_size_per_second()

    volume_preservation()
    reversibility()

    # gaussian_effective_sample_size()

if __name__ == '__main__':
    main()
