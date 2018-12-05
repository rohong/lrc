#!/usr/bin/env python3
"""
Plot coherent power spectra
"""
import matplotlib.pyplot as plt
import numpy as np
# from numpy.fft import fftshift as fsh
import seaborn as sns
from scipy import signal
from raw_data_async import get_dbs
import time as sys_time

# import _data
# import cmocean.cm as cmo
plt.switch_backend('Agg')
sns.set(style='ticks', palette='Set2')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})


# %%
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


@log
def calc_coh_pwr(shot, t1, t2):
    fname = f"../raw_data/DBS_{shot}.nc"
    dbs, t_dbs = get_dbs(shot, (t1, t2))
    print(f"=== Loaded {fname}")

    da = dbs.real
    db = dbs.imag
    fs = t_dbs.size / (t_dbs[-1] - t_dbs[0])
    nperseg = int(fs * 2)
    nfft = pow2(nperseg)
    noverlap = nperseg // 2

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


@log
def plot_2d(spec, freq, t, shot):
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), sharex='col', sharey='all')
    for i, ax in enumerate(axs.flatten(order='F')):
        print(f"=== Channel {i + 1}")
        ax.pcolormesh(t, freq, spec[i, :, :], cmap='CMRmap')
        ax.text(t[50], 2000, f'Ch{i + 1}', color='white', fontsize=11)
        if i == 4:
            ax.set_title(f'#{shot}', x=0.8, fontsize=9)
        if i < 4:
            ax.set(ylabel='f (kHz)')
        if i % 4 == 3:
            ax.set(xlabel='time (ms)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.1)
    fig.savefig(f'../fig/coh_pwr_{shot}.png', transparent=True)
    plt.close()


@log
def plot_1d(spec, freq, shot):
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), sharex='col',
                            sharey='all')
    fidx = freq > 20
    for i, ax in enumerate(axs.flatten(order='F')):
        print(f"=== Channel {i + 1}")
        ax.semilogy(freq[fidx], spec[i, fidx, :].mean(-1), lw=0.8)
        ax.text(0.9, 0.9, f'Ch{i + 1}', color='k', fontsize=11,
                transform=ax.transAxes)
        sns.despine()
        if i == 4:
            ax.set_title(f'#{shot}', x=0.8, fontsize=9)
        if i % 4 == 3:
            ax.set(xlabel='f (kHz)')

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0, wspace=0.1)
    fig.savefig(f'../fig/coh_pwr_{shot}_1d.pdf', transparent=True)
    plt.close()


# %%
if __name__ == "__main__":
    shot = 150136
    t1, t2 = 0, 5e3
    coh_pwr, freq, t = calc_coh_pwr(shot, t1, t2)

    # %%
    # spec = signal.medfilt(np.log10(coh_pwr), [1, 3, 1])
    # plot_2d(spec, freq, t, shot)
    # %%
    plot_1d(coh_pwr, freq, shot)
