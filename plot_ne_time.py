#!/usr/bin/env python3
"""
Plot ne rms time series based on coherent power spectra
Plot conditional sampling results
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns
from scipy import signal, interpolate
from raw_data_async import get_dbs
import time as sys_time
import os.path
import _data, numba

# import cmocean.cm as cmo
# plt.switch_backend('Agg')
sns.set(style='ticks', palette='Set2')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})


def log(func):
    def func_dec(*args, **kwargs):
        print(f"=== Start {func.__name__} ===")
        time_start = sys_time.time()
        r = func(*args, **kwargs)
        time_cost = sys_time.time() - time_start
        print(f"=== {func.__name__} done, cost {time_cost:.0f} seconds ===")
        return r

    return func_dec


def pow2(x):
    return np.power(2, np.ceil(np.log2(x)))


# %%
# @log
def calc_quad(shot, t1, t2, overlap=0.99):
    fname = f"../raw_data/DBS_{shot}.nc"
    dbs, t_dbs = get_dbs(shot, (t1, t2))
    print(f"=== Loaded {fname}")

    fs = t_dbs.size / (t_dbs[-1] - t_dbs[0])
    nperseg = int(1000 * 1)
    nfft = pow2(nperseg) * 2
    noverlap = int(nperseg * overlap)

    freq, time, quad = signal.spectrogram(dbs, nperseg=nperseg, nfft=nfft,
                                          noverlap=noverlap, fs=fs,
                                          return_onesided=False)
    time += t1
    print(f'=== Calculated quadrature shot {shot}.')
    return quad, freq, time


# @log
def calc_coh_pwr(shot, t1, t2, overlap=0.99):
    fname = f"../raw_data/DBS_{shot}.nc"
    dbs, t_dbs = get_dbs(shot, (t1, t2))
    print(f"=== Loaded {fname}")

    da = dbs.real
    db = dbs.imag
    fs = t_dbs.size / (t_dbs[-1] - t_dbs[0])
    nperseg = int(1000 * 1)
    nfft = pow2(nperseg) * 2
    noverlap = int(nperseg * overlap)

    freq, t, spec_a = signal.spectrogram(da, nperseg=nperseg, nfft=nfft,
                                         noverlap=noverlap, fs=fs,
                                         mode='complex')
    _, _, spec_b = signal.spectrogram(db, nperseg=nperseg, nfft=nfft,
                                      noverlap=noverlap, fs=fs,
                                      mode='complex')

    coh = np.abs(spec_a * np.conj(spec_b)) ** 2 / np.abs(spec_a) ** 2 / np.abs(
        spec_b) ** 2

    t += t1
    print(f'=== Calculated coherent power spectra shot {shot}.')
    return coh * np.abs(spec_a) ** 2, freq, t


# @log
def plot_time(ne, time, shot, x1=2500, x2=3000):
    tidx = np.logical_and(time > x1, time < x2)
    y = (ne - np.mean(ne, axis=-1, keepdims=True)) / np.std(ne, axis=-1,
                                                            keepdims=True)
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='all', figsize=(9, 6))
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.plot(time[tidx], y[i, tidx], lw=0.8)
        for l in [0, 1, 2]:
            ax.axhline(l, ls='--', c='gray', lw=0.4)
        ax.text(0.9, 0.85, f'Ch{i + 1}', transform=ax.transAxes)
        if i % 4 == 3:
            ax.set_label('time (ms)')
        if i == 4:
            ax.set_title(f'#{shot}', x=0.8, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0)
    fig.savefig(f"../fig/ne_time_{shot}_{x1}-{x2}.pdf", transparent=True)
    plt.close()


@log
def calc_ne_rms(shot, t1, t2, kind='coh'):
    if kind == 'coh':
        spec, frq, t = calc_coh_pwr(shot, t1, t2)
    elif kind == 'quad':
        spec, frq, t = calc_quad(shot, t1, t2)
    else:
        print("Error Kind of Calculation!")
        return None

    spec_m = signal.savgol_filter(spec.mean(-1), 31, 3)

    ne = np.empty((spec.shape[0], spec.shape[-1]))
    for i in range(8):
        if i == 3 or i == 4:
            fidx = np.logical_or(frq > 700, frq < -1600)
        elif i > 4:
            fidx = abs(frq) > 150
        else:
            idx = frq > -5000
            ind = np.argmax(spec_m[i, idx])
            fpk = (frq[idx])[ind]
            delf = 700
            fidx = np.logical_and.reduce(
                (frq > fpk - delf, frq < fpk + delf))
            print(f"f{i + 1}-peak = {fpk:0f} kHz")
        ne[i, :] = np.sum(spec[i, fidx, :], axis=0)

    return ne, t


@log
def save_ne_rms(ne, time, filepath):
    chan = np.arange(1, 9)
    ds = xr.Dataset({'ne': (('chan', 'time'), ne)},
                    coords={'chan': chan, 'time': time})
    ds.to_netcdf(filepath)
    print(f"Saved data to {filepath}")


def cond_avg(yy, idx, n_lag):
    tlag = np.arange(-n_lag, n_lag) / fs1
    ne_pk = np.empty((8, tlag.size), dtype=float)
    for i in range(8):
        ne_pk[i, :] = np.mean(
            np.array([yy[i, pk - n_lag:pk + n_lag] for pk in idx]), axis=0)
    return ne_pk, tlag


def plot_cond_samp_time(yy, time, idx, figpath):
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='all', figsize=[8, 8])
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.plot(time, yy[i, :], label=f'Ch{i + 1}', lw=0.5)
        ax.plot(time[idx], yy[i, idx], 'p', alpha=0.5, markersize=4)
        ax.axhline(0, ls=':', c='gray')
        ax.set(xlim=(2540, 2575), ylim=(np.min(yy.flat), 5))
        sns.despine()
        if i % 4 == 3:
            ax.set_xlabel(r'time (ms)')
        if i != ch:
            ax.text(0.88, 0.9, f'Ch{i + 1}', transform=ax.transAxes)
        else:
            ax.axhline(y_thr, ls='-', c='k', lw=0.5)
            ax.text(0.75, 0.9, f'Ch{i + 1} reference', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(figpath, transparent=True)
    plt.show()


def plot_cond_avg(ne_pk, tlag, figpath):
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='row', figsize=[8, 8])
    for i, ax in enumerate(axs.flatten(order='F')):
        yplt = ne_pk[i, :] / ne_pk[i, :].max()
        x_intp = np.linspace(-50, 50, 5000)
        f_intp = interpolate.interp1d(tlag * 1e3, yplt, kind='cubic')
        y_intp = f_intp(x_intp)
        ipk = np.argmax(y_intp)

        ax.plot(tlag * 1e3, yplt, label=f'Ch{i + 1}')
        ax.axvline(x_intp[ipk], c='C3', ls='--')
        ax.axvline(0, ls=':', c='gray', lw=0.8)
        ax.axhline(0, ls=':', c='gray', lw=0.8)
        if i != ch:
            ax.text(0.65, 0.85,
                    f'Ch{i + 1} lag={x_intp[ipk]:.0f}' + r' $\mu s$',
                    transform=ax.transAxes)
        else:
            ax.text(0.65, 0.85, f'Ch{i + 1} reference',
                    transform=ax.transAxes)
        ax.set(ylim=[-0.3, 1.1])
        sns.despine()
        if i % 4 == 3:
            ax.set_xlabel(r'$\Delta t \,(\mu s)$ ')
    plt.tight_layout()
    plt.savefig(figpath, transparent=True)
    plt.show()


# %%
if __name__ == "__main__":
    shot = 150136
    # t1, t2 = 2000, 3200
    t1, t2 = 2450, 2650
    readne = True
    # readne = False
    # %%
    filepath = f'../proc_data/fft_ne_{shot}_{t1}-{t2}_coh.nc'

    if os.path.isfile(filepath) and readne:
        ds = xr.open_dataset(filepath)
        ne = ds['ne'].values
        time = ds['time'].values
        print("Loaded processed nc file.")
    else:
        ne, time = calc_ne_rms(shot, t1, t2, kind='quad')
        save_ne_rms(ne, time, filepath)

    # %%
    # plot_time(ne1, time, shot, x1=2570, x2=2575)
    # %%
    # Conditoinal sampling
    sm_win = 301
    n_lag = 100
    fs1 = len(time) / (time[-1] - time[0])

    ece, tece = _data.get_mds('ece5', shot)
    ece1 = signal.medfilt(np.interp(time, tece, ece), sm_win)
    dTe = signal.medfilt(abs(np.gradient(ece1)), sm_win)
    dTe /= dTe.max()

    ch = 2
    ne_norm = (ne - np.mean(
        ne, axis=-1, keepdims=True)) / np.std(ne, axis=-1, keepdims=True)
    y = ne_norm[ch, :]
    y = (y - y.mean()) / y.std()
    y_thr = 1
    dy = max(int(fs1 * 0.1), 10)

    idx = np.array([i for i in range(n_lag, len(y) - n_lag) if y[i] > y_thr
                    if y[i] == np.max(y[i - dy:i + dy]) if
                    dTe[i] < 0.05])
    print(f"{idx.size} samples found")

    yy = ne_norm
    figpath = f"../fig/conditional_sampling.pdf"
    plot_cond_samp_time(yy, time, idx, figpath)

    # plt.figure()
    # plt.plot(time, y, lw=0.5)
    # plt.plot(time[idx], y[idx], '+', alpha=0.5)
    # plt.plot(time, dTe / dTe.max() * 10, '.')
    # plt.plot(time, ece1 / ece1.max() * 8)
    # plt.show()

    # %%

    ne_pk, tlag = cond_avg(yy, idx, n_lag)

    figpath = f'../fig/conditonal_average.pdf'
    plot_cond_avg(ne_pk, tlag, figpath)
