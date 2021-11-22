import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hmc import summarize


def euclidean_samples():
    num_samples = [1000000]
    euclid = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*num-samples-{}-*euclidean*'.format(ns))))
        for f in fns:
            ss = f.split('-step-size-')[1].split('-')[0]
            ss = float(ss)
            with open(f, 'rb') as g:
                d[ss] = pickle.load(g)
        euclid[ns] = d
    return euclid

def softabs_samples():
    num_samples = [1000000]
    rmn = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*-step-size-0.2*num-samples-{}-*softabs*'.format(ns))))
        for f in fns:
            t = f.split('-thresh-')[1].split('-m')[0]
            t = float(t)
            with open(f, 'rb') as g:
                d[t] = pickle.load(g)
        rmn[ns] = d
    return rmn

def effective_sample_size():
    euclid = euclidean_samples()[1000000]
    rmn = softabs_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)

    labels = ['Euclid. {}'.format(t) for t in ekeys] + ['Thresh. {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    num_breaks = 20
    ess = {}
    for t in ekeys:
        breaks = np.split(euclid[t]['samples'], num_breaks, axis=0)
        k = 'euclid-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min()
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        breaks = np.split(rmn[t]['samples'], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min()
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 5, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Minimum ESS')
    ax.grid(linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'minimum-ess.pdf'))

def effective_sample_size_per_second():
    euclid = euclidean_samples()[1000000]
    rmn = softabs_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)

    labels = ['Euclid. {}'.format(t) for t in ekeys] + ['Thresh. {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    num_breaks = 20
    ess = {}
    for t in ekeys:
        breaks = np.split(euclid[t]['samples'], num_breaks, axis=0)
        k = 'euclid-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (euclid[t]['time'] / num_breaks)
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        breaks = np.split(rmn[t]['samples'], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (rmn[t]['time'] / num_breaks)
            ess[k].append(m)

    ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 5, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Min. ESS / Sec.', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'minimum-ess-per-second.pdf'))

def box_plot(ax, data, positions, offset, color):
    loc = positions + offset
    bp = ax.boxplot(data, notch=True, patch_artist=True, positions=loc)
    for patch in bp['boxes']:
        patch.set(facecolor=color)
    return bp

def mmd():
    euclid = euclidean_samples()[1000000]
    rmn = softabs_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    num_thresholds = len(rkeys)
    thresholds = np.array(rkeys)

    emmd = np.log10(np.abs([euclid[k]['mmd'] for k in ekeys]))
    rmmd = np.log10(np.abs([rmn[k]['mmd'] for k in rkeys]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rmmd, '.-')
    ls = ['-', '--', ':', '-.']
    for i, v in enumerate(emmd):
        ax.axhline(v, color='k', linestyle=ls[i], label='Euclid. {:.3f}'.format(ekeys[i]))
    ax.legend(fontsize=24)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, num_thresholds))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(linestyle=':')
    ax.set_xlabel(r'$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel(r'$\log_{10} |\mathrm{MMD}^2|$ Estimate', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'mmd.pdf'))

def kolmogorov_smirnov():
    euclid = euclidean_samples()[1000000]
    rmn = softabs_samples()[1000000]
    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)

    labels = ['Euclid. {}'.format(t) for t in ekeys] + ['Thresh. {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    ess = {}
    for t in ekeys:
        k = 'euclid-{}'.format(t)
        ess[k] = np.log10(euclid[t]['ks'])

    vpa = ax.violinplot([ess[k] for k in ess.keys()], showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        k = 'rmn-{}'.format(t)
        ess[k] = np.log10(rmn[t]['ks'])

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 5, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=16)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('KS Statistic', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'kolmogorov-smirnov.pdf'))

def wasserstein_sliced():
    euclid = euclidean_samples()[1000000]
    rmn = softabs_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    num_thresholds = len(rkeys)
    thresholds = np.array(rkeys)

    esw = np.log10(np.abs(np.array([euclid[k]['sw'] for k in ekeys])))
    rsw = np.log10(np.abs(np.array([rmn[k]['sw'] for k in rkeys])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rsw, '.-')
    for v in esw:
        ax.axhline(v, color='k')
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, num_thresholds))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(linestyle=':')
    ax.set_xlabel(r'$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel(r'$\log_{10}$ Sliced Wasserstein', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'sw.pdf'))

def volume_preservation():
    euclid = euclidean_samples()
    rmn = softabs_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [rmn[1000000][t]['jacdet'][1e-5] for t in thresholds]
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
    ax.set_ylabel('$\log_{10}$ Volume Preservation Error', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'jacobian-determinant.pdf'))

    perturb = sorted(rmn[1000000][1e-9]['jacdet'].keys())
    num_perturb = len(perturb)
    dat = [rmn[1000000][1e-9]['jacdet'][p] for p in perturb]
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
    euclid = euclidean_samples()
    rmn = softabs_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [rmn[1000000][t]['absrev'] for t in thresholds]
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

    dat = [rmn[1000000][t]['relrev'] for t in thresholds]
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
    euclid = euclidean_samples()
    rmn = softabs_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [np.log10(rmn[1000000][t]['nfp_mom']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    dat = [_[np.random.permutation(len(_))[:10000]] for _ in dat]
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

def position_fixed_point():
    euclid = euclidean_samples()
    rmn = softabs_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)
    dat = [np.log10(rmn[1000000][t]['nfp_pos']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    dat = [_[np.random.permutation(len(_))[:10000]] for _ in dat]
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

def main():
    wasserstein_sliced()
    kolmogorov_smirnov()
    mmd()

    volume_preservation()
    reversibility()

    effective_sample_size()
    effective_sample_size_per_second()

    momentum_fixed_point()
    position_fixed_point()

if __name__ == '__main__':
    main()
