import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hmc import summarize


def euclidean_samples():
    num_samples = [1000, 10000, 100000, 1000000]
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

def riemannian_samples(newton_momentum=False, newton_position=False):
    num_samples = [1000, 10000, 100000, 1000000]
    rmn = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*num-steps-25-*num-samples-{}-*riemannian*-partial-momentum-0.0-*newton-momentum-{}*newton-position-{}*'.format(ns, newton_momentum, newton_position))))
        for f in fns:
            t = f.split('-thresh-')[1].split('-m')[0]
            t = float(t)
            with open(f, 'rb') as g:
                d[t] = pickle.load(g)
        rmn[ns] = d
    return rmn

def implicit_samples():
    num_samples = [1000, 10000, 100000, 1000000]
    imp = {}
    for ns in num_samples:
        d = {}
        fns = sorted(glob.glob(os.path.join('samples', '*step-size-0.1*num-steps-25-*num-samples-{}-*implicit*-partial-momentum-0.0-*'.format(ns))))
        for f in fns:
            t = f.split('-thresh-')[1].split('-m')[0]
            t = float(t)
            with open(f, 'rb') as g:
                d[t] = pickle.load(g)
        imp[ns] = d
    return imp

def mmd():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)
    num_thresholds = len(rkeys)
    thresholds = np.array(rkeys)

    emmd = np.log10(np.abs([euclid[k]['mmd'] for k in ekeys]))
    rmmd = np.log10(np.abs([rmn[k]['mmd'] for k in rkeys]))
    immd = np.log10(np.abs([imp[k]['mmd'] for k in ikeys]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rmmd, '.-', color='tab:orange')
    ax.plot(immd, '.-', color='tab:green')
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

def wasserstein_sliced():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)
    num_thresholds = len(rkeys)
    thresholds = np.array(rkeys)

    esw = np.log10(np.abs(np.array([euclid[k]['sw'] for k in ekeys])))
    isw = np.log10(np.abs(np.array([imp[k]['sw'] for k in ikeys])))
    rsw = np.log10(np.abs(np.array([rmn[k]['sw'] for k in rkeys])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rsw, '.-', color='tab:orange')
    ax.plot(isw, '.-', color='tab:green')
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

def biau():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)
    num_thresholds = len(rkeys)
    thresholds = np.array(rkeys)

    ebiau = np.log10(np.abs([euclid[k]['biau'] for k in ekeys]))
    rbiau = np.log10(np.abs([rmn[k]['biau'] for k in rkeys]))
    ibiau = np.log10(np.abs([imp[k]['biau'] for k in ikeys]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rbiau, '.-', color='tab:orange')
    ax.plot(ibiau, '.-', color='tab:green')
    ls = ['-', '--', ':', '-.']
    for i, v in enumerate(ebiau):
        ax.axhline(v, color='k', linestyle=ls[i], label='Euclid. {:.3f}'.format(ekeys[i]))
    ax.legend(fontsize=24)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, num_thresholds))
    ax.set_xticklabels(['{:.0f}'.format(np.log10(t)) for t in thresholds], fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(linestyle=':')
    ax.set_xlabel(r'$\log_{10}$ Threshold', fontsize=30)
    ax.set_ylabel(r'Discretized $L_1$ Distance', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'biau.pdf'))

def effective_sample_size():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)

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

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        breaks = np.split(imp[t]['samples'], num_breaks, axis=0)
        k = 'imp-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min()
            ess[k].append(m)

    vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

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
    ax.legend([vpb["bodies"][0], vpc["bodies"][0]], [r'Riemann. (G.L.F. $\epsilon=0.04$)', r'Riemann. (I.M. $\epsilon=0.1$)'], fontsize=16)

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'minimum-ess.pdf'))

def effective_sample_size_per_second():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    nm_rmn = riemannian_samples(True)[1000000]
    nb_rmn = riemannian_samples(True, True)[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    nrkeys = sorted(nm_rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)

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

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        breaks = np.split(imp[t]['samples'], num_breaks, axis=0)
        k = 'imp-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (imp[t]['time'] / num_breaks)
            ess[k].append(m)

    vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

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
    ax.legend([vpb["bodies"][0], vpc["bodies"][0]], [r'Riemann. (G.L.F. $\epsilon=0.04$)', r'Riemann. (I.M. $\epsilon=0.1$)'], fontsize=16)

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'minimum-ess-per-second.pdf'))

    labels = ['Thresh. {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    num_breaks = 20
    ess = {}
    for t in rkeys:
        breaks = np.split(rmn[t]['samples'], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (rmn[t]['time'] / num_breaks)
            ess[k].append(m)

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in nrkeys:
        breaks = np.split(nm_rmn[t]['samples'], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (nm_rmn[t]['time'] / num_breaks)
            ess[k].append(m)

    vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(nrkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in nrkeys:
        breaks = np.split(nb_rmn[t]['samples'], num_breaks, axis=0)
        k = 'rmn-{}'.format(t)
        ess[k] = []
        for i, b in enumerate(breaks):
            metrics = summarize(b)
            m = metrics['ess'].min() / (nb_rmn[t]['time'] / num_breaks)
            ess[k].append(m)

    vpd = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(nrkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)


    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel('')
    ax.set_ylabel('Min. ESS / Sec.', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(linestyle=':')
    ax.legend([vpb["bodies"][0], vpc["bodies"][0], vpd["bodies"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=16, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'minimum-ess-per-second-vs-newton.pdf'))

def kolmogorov_smirnov():
    euclid = euclidean_samples()[1000000]
    rmn = riemannian_samples()[1000000]
    nm_rmn = riemannian_samples(True)[1000000]
    nb_rmn = riemannian_samples(True, True)[1000000]
    imp = implicit_samples()[1000000]

    ekeys = sorted(euclid.keys(), reverse=False)
    rkeys = sorted(rmn.keys(), reverse=False)
    ikeys = sorted(imp.keys(), reverse=False)

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

    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in ikeys:
        k = 'imp-{}'.format(t)
        ess[k] = np.log10(imp[t]['ks'])

    vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(ikeys)) + 2, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=16)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.axvline(len(ekeys) + 0.5, color='black', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('KS Statistic', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.grid(linestyle=':')
    ax.legend([vpb["bodies"][0], vpc["bodies"][0]], [r'Riemann. (G.L.F.)', r'Riemann. (I.M.)'], fontsize=16)

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'kolmogorov-smirnov.pdf'))

    labels = ['Thresh. {:.0e}'.format(t) for t in rkeys]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ess = {}
    for t in rkeys:
        k = 'rmn-{}'.format(t)
        ess[k] = np.log10(rmn[t]['ks'])
    vpb = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        k = 'rmn-{}'.format(t)
        ess[k] = np.log10(nm_rmn[t]['ks'])
    vpc = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

    ess = {}
    for t in rkeys:
        k = 'rmn-{}'.format(t)
        ess[k] = np.log10(nb_rmn[t]['ks'])
    vpd = ax.violinplot([ess[k] for k in ess.keys()], positions=np.arange(len(rkeys)) + 1, showmeans=True, showmedians=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(['' for l in labels])
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=24)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('')
    ax.set_ylabel('KS Statistic', fontsize=30)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim((-3, 3))
    ax.grid(linestyle=':')
    ax.legend([vpb["bodies"][0], vpc["bodies"][0], vpd["bodies"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=20, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join('images', 'kolmogorov-smirnov-vs-newton.pdf'))

def volume_preservation():
    euclid = euclidean_samples()
    rmn = riemannian_samples()
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
    ax.set_ylabel('$\log_{10}$ Vol. Pres. Err.', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'jacobian-determinant.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='tab:blue')

    nm_rmn = riemannian_samples(True)
    dat = [nm_rmn[1000000][t]['jacdet'][1e-5] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nm_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nm_bp['boxes']:
        patch.set(facecolor='tab:red')

    nb_rmn = riemannian_samples(True, True)
    dat = [nb_rmn[1000000][t]['jacdet'][1e-5] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    flierprops = dict(markeredgecolor='tab:green')
    nb_bp = ax.boxplot(dat, notch=True, patch_artist=True, flierprops=flierprops)
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
    ax.legend([bp["boxes"][0], nm_bp["boxes"][0], nb_bp["boxes"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=20, loc='upper left')
    ax.set_ylim((-16, 30))
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'jacobian-determinant-vs-newton.pdf'))

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
    rmn = riemannian_samples()
    imp = implicit_samples()
    num_thresholds = 9
    thresholds = np.logspace(-num_thresholds, -1, num_thresholds)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    dat = [rmn[1000000][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
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
    dat = [nm_rmn[1000000][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    nm_bp = ax.boxplot(dat, notch=True, patch_artist=True)
    for patch in nm_bp['boxes']:
        patch.set(facecolor='tab:red')

    nb_rmn = riemannian_samples(True, True)
    dat = [nb_rmn[1000000][t]['absrev'] for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    flierprops = dict(markeredgecolor='tab:green')
    nb_bp = ax.boxplot(dat, notch=True, patch_artist=True, flierprops=flierprops)
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
    ax.legend([bp["boxes"][0], nm_bp["boxes"][0], nb_bp["boxes"][0]], [r'Fixed Point', r'Newton (Mom.)', r'Newton (Mom. and Pos.)'], fontsize=20, loc='upper left')
    ax.set_ylim((-16, 20))
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'absolute-reversibility-vs-newton.pdf'))

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
    rmn = riemannian_samples()
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
    ax.set_ylabel('$\log_{10}$ Mom. Fixed Point', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-momentum.pdf'))

    nrmn = riemannian_samples(True)
    dat = [np.log10(rmn[1000000][t]['nfp_mom']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    mean = np.array([np.mean(_) for _ in dat])
    std = np.array([np.std(_) / np.sqrt(len(_)) for _ in dat])
    ndat = [np.log10(nrmn[1000000][t]['nfp_mom']) if t in nrmn[1000000] else np.array([np.nan]) for t in thresholds]
    ndat = [_[~np.isnan(_)] for _ in ndat]
    nmean = np.array([np.mean(_) for _ in ndat])
    nstd = np.array([np.std(_) / np.sqrt(len(_)) for _ in ndat])

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
    ax.set_ylim((0.0, 2))
    ax.legend(fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-momentum-vs-newton.pdf'))

def position_fixed_point():
    euclid = euclidean_samples()
    rmn = riemannian_samples()
    nrmn = riemannian_samples(True, True)

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
    ax.set_ylabel('$\log_{10}$ Pos. Fixed Point', fontsize=30)
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'num-fixed-point-position.pdf'))

    dat = [np.log10(rmn[1000000][t]['nfp_pos']) for t in thresholds]
    dat = [_[~np.isnan(_)] for _ in dat]
    mean = np.array([np.mean(_) for _ in dat])
    std = np.array([np.std(_) for _ in dat])
    ndat = [np.log10(nrmn[1000000][t]['nfp_pos']) for t in thresholds]
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
    kolmogorov_smirnov()
    wasserstein_sliced()
    biau()
    mmd()

    volume_preservation()
    reversibility()

    momentum_fixed_point()
    position_fixed_point()

    effective_sample_size_per_second()
    effective_sample_size()


if __name__ == '__main__':
    main()
