#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Check DBS data
"""
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from raw_data import get_dbs
import scipy.signal as sps
import os

# import _axes_prop
# plt.switch_backend('Agg')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})
sns.set(style='ticks', palette='Set2')
# %%

if __name__ == "__main__":
    shot = 150136
    fs = 5e3
    t1, t2 = 2500, 2700
    nperseg = 1024
    nfft = nperseg * 2
    overlap = 0.9

    data, t = get_dbs(shot, (t1, t2))
    print("Read data from file.")

    da = data.real
    db = data.imag
    noverlap = int(nperseg * overlap)
    freq, time, Sa = sps.spectrogram(da, nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap, fs=fs)
    freq, coh = sps.coherence(da, db, nperseg=nperseg, nfft=nfft,
                              noverlap=noverlap, fs=fs)
    # print(Sx.shape, freq.shape)
    Scoh = Sa * coh[:, :, None]

    # %%
    filepath = f'../proc_data/fft_coh_pwr_{shot}_{t1}-{t2}.nc'
    if not os.path.isfile(filepath):
        chan = np.arange(1, 9)
        ds = xr.Dataset({'Scoh': (('chan', 'freq', 'time'), Scoh)},
                        coords={'chan': chan, 'freq': freq, 'time': time + t1})
        ds.to_netcdf(filepath)

    # %%
    # fig, axs = plt.subplots(4, 2, sharex='col', figsize=[8, 6])
    # print("Plotting.")
    # for i, ax in enumerate(axs.flatten(order='F')):
    #     ax.pcolormesh(time + t1, freq, np.log10(Scoh[i, :, :]),
    #                   cmap='viridis')
    #     ax.set(ylabel='f (kHz)', title=f'ch{i+1}')
    # plt.tight_layout()
    # fig.savefig(f'../fig/coherent_power_spectra.png', dpi=300,
    #             transparent=True)
    # plt.show()
    # %%
    fig, axs = plt.subplots(4, 2, sharex='col', figsize=[8, 6])
    for i, ax in enumerate(axs.flatten(order='F')):
        S_mean = np.mean(Scoh[i, :, :], axis=-1)
        ax.plot(freq, S_mean, lw=0.8)
        ax.plot(freq, sps.savgol_filter(S_mean, 41, 3), lw=1)
        ax.set(title=f'ch{i+1}')
        if i % 4 == 3:
            ax.set(xlabel='f (kHz)')
    plt.tight_layout()
    fig.savefig(f'../coherent_power_spectra_{shot}_1d.pdf', transparent=True)
    plt.show()
