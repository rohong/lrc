#!/usr/bin/env python3
"""
Check DBS data for density peak envelope
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift as fsh
import seaborn as sns
from scipy import signal
from raw_data_async import get_dbs

# import _data
# import cmocean.cm as cmo
plt.switch_backend('agg')
sns.set(style='ticks', palette='Set2')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})


# %%
def pow2(x):
    return np.power(2, np.ceil(np.log2(x)))


def calc_quad(shot, t1, t2):
    fname = f"../raw_data/DBS_{shot}.nc"
    dbs, t_dbs = get_dbs(shot, (t1, t2))
    print(f"=== Loaded {fname}")

    fs = t_dbs.size / (t_dbs[-1] - t_dbs[0])
    nperseg = int(fs * 2)
    nfft = pow2(nperseg)
    freq, time, quad = signal.spectrogram(dbs, nperseg=nperseg, nfft=nfft,
                                          noverlap=nperseg // 2, fs=fs,
                                          return_onesided=False)
    time += t1
    print(f'=== Calculated quadrature shot {shot}.')
    return quad, freq, time


def test_cmap(spec, freq, time, cmap):
    plt.figure()
    plt.pcolormesh(time, fsh(freq), fsh(spec, axes=0), cmap=cmap)
    plt.title(f"{cmap}")
    plt.show()


# %%

if __name__ == "__main__":
    shot = 150136
    t1, t2 = 0, 5e3
    quad, freq, time = calc_quad(shot, t1, t2)
    # # %%
    # spec = signal.medfilt(np.log10(quad[0, :, :]))
    # del quad
    # maps = ['cmo.thermal', 'cmo.haline', 'cmo.ice', 'gist_heat', 'inferno',
    #         'viridis', 'CMRmap', 'gist_earth', 'Spectral', 'nipy_spectral',
    #         'cubehelix', 'gist_ncar', 'rainbow']
    # for icmap in maps:
    #     test_cmap(spec, freq, time, cmap=icmap)

    # %%
    spec = signal.medfilt(np.log10(quad), [1, 5, 1])

    print(f"=== Start plotting {shot} quadrature.")
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), sharex='col', sharey='all')
    for i, ax in enumerate(axs.flatten(order='F')):
        print(f"=== Channel {i + 1}")
        ax.pcolormesh(time, fsh(freq), fsh(spec[i, :, :], axes=0),
                      cmap='CMRmap')
        ax.text(time[50], 2000, f'Ch{i + 1}', color='white', fontsize=11)
        if i == 4:
            ax.set(title=f'{shot}', x=0.7, fontsize=9)
        if i < 4:
            ax.set(ylabel='f (kHz)')
        if i % 4 == 3:
            ax.set(xlabel='time (ms)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.savefig(f'../fig/quad_{shot}.png', transparent=True, dpi=150)
    plt.close()
