#!/usr/bin/env python3
"""
Check DBS data for density peak envelope
"""
import _data
import matplotlib.pyplot as plt
import numba as nba
import numpy as np
import seaborn as sns
import xarray as xr
from scipy.signal import spectrogram, coherence
from raw_data import get_dbs
import os.path, asyncio

# import _axes_prop
# plt.switch_backend('Agg')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})
sns.set(style='ticks', palette='Set2')


# %%
@nba.jit(cache=True)
def calc_ne_rms(data: 'dbs array', t: 'dbs time', nfft: int = 1024 * 2,
                overlap=0.9, f1: 'freq low' = 1e2, f2: 'freq high' = 1.5e3
                ) -> ('ne', 'time'):
    """Calculate ne rms level by integrating spectrogram along freq axis"""
    noverlap = int(nfft * overlap)
    freq, time, Sx = spectrogram(data, nperseg=nfft, noverlap=noverlap, fs=5e3,
                                 return_onesided=False)
    # print(Sx.shape, freq.shape)
    fidx = np.logical_and(freq > f1, freq < f2)
    ne = np.sum(Sx[:, fidx, :], axis=1)
    return ne, time + t[0]


@nba.jit(cache=True)
def calc_coh_all(ne: 'density array', nperseg: int = 1024 * 2,
                 nfft: int = 1024 * 4, overlap=0.9,
                 iref: 'ref channel' = 2) -> ('coherence', 'freq: array'):
    """Calculate coherence between each channels"""
    nem = (ne - np.mean(ne, axis=-1, keepdims=True)) / \
          np.std(ne, axis=-1, keepdims=True)
    xcoh = np.empty((nem.shape[0], nfft // 2 + 1))
    noverlap = int(nperseg * overlap)
    for i in range(ne.shape[0]):
        freq, xcoh[i, :] = coherence(nem[iref, :], nem[i, :], fs=5e3,
                                     nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap)
    return xcoh, freq


def plot_ne_time(ne, time) -> None:
    """Plot ne fluctuation time series"""
    fig, axs = plt.subplots(4, 2, sharex='col', figsize=[8, 6])
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.plot(time, ne[i, :], lw=0.5)
        ax.set_ylabel(f'n{i+1}')
    plt.tight_layout()
    plt.savefig('../fig/ne_rms_fft_time.pdf', transparent=True)
    # plt.show()


def plot_ne_coh_all(xcoh, freq, iref=2) -> None:
    """Plot coherence between different channels."""
    plt.figure()
    for i in range(xcoh.shape[0]):
        if i != iref:
            plt.plot(freq, xcoh[i, :], lw=1, label=f'ch{iref+1}-{i+1}')
    plt.legend(loc=0)
    plt.xlabel('f (kHz)')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('../fig/coherence_rms_fft.pdf', transparent=True)
    plt.show()


def plot_ne_coh_pair(xcoh: 'array', freq: 'freq',
                     chans: ('channel 1', 'channel 2')) -> None:
    ch1, ch2 = chans
    plt.figure()
    plt.semilogx(freq, xcoh, label=f'ch{ch1}-{ch2}')
    plt.legend(loc=0)
    # plt.xlim(1, 000)
    plt.savefig(f'../fig/coherence_ch{ch1}-{ch2}.pdf')
    plt.show()


# %%
if __name__ == "__main__":
    shot = 150136
    fs = 5e3
    iref = 0
    t1, t2 = 2400, 2700
    f1, f2 = 200, 1.5e3
    readne = True

    filepath = f'../proc_data/{shot}_fft_ne_{t1}-{t2}.nc'
    if os.path.isfile(filepath) and readne:
        ds = xr.open_dataset(filepath)
        ne = ds['ne'].values
        time = ds['time'].values
    else:
        data, t = get_dbs(shot, (t1, t2))
        ne, time = calc_ne_rms(data, t, overlap=0.95)
        chan = np.arange(1, 9)
        ds = xr.Dataset({'ne': (('chan', 'time'), ne)},
                        coords={'chan': chan, 'time': time})
        ds.to_netcdf(filepath)

    ece, tece = _data.get_mds('ece5', shot)
    ece1 = np.interp(time, tece, ece)
    ind = ece1 <= 0.8
    ne1, time1 = ne[:, ind], time[ind]
    print(ne1.shape)

    nfft = 1024 * 4
    # plotall = True
    plotall = False
    if plotall:
        xcoh, freq = calc_coh_all(ne1, nfft=nfft, iref=iref)
        # plot_ne_time(ne1, time1)
        plot_ne_coh_all(xcoh, freq, iref=iref)
    else:
        ch1, ch2 = 1, 8
        x, y = ne1[ch1 - 1, :], ne1[ch2 - 1, :]
        x, y = ne[ch1 - 1, :], ne[ch2 - 1, :]
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        freq, xcoh = coherence(x, y, fs=fs, nperseg=nfft // 2, nfft=nfft,
                               noverlap=int(0.9 * nfft // 2))
        plot_ne_coh_pair(xcoh, freq, (ch1, ch2))
