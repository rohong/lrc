#!/usr/bin/env python3
# pylint: disable=invalid-name
'''
Get ECE time series of different channels
Compare with modes' spectrogram
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, savgol_filter
# import _axes_prop
import _data
plt.switch_backend('Agg')
# plt.switch_backend('pdf')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4]})
sns.set(style='ticks', palette='Set2')


# %%
def get_ece(shot):
    '''get ece data'''
    ece = np.array([])
    chs = []
    for ch in range(3, 32, 5):
        pname = f'ece{ch}'
        chs.append(pname)
        y, t = _data.get_mds(pname, shot)
        if ece.size == 0:
            ece = y
        else:
            ece = np.vstack((ece, y))
    return ece, t, chs


def get_bdot_spec(shot):
    '''get b4 data'''
    bdot, t = _data.get_mds('b1', shot)
    nfft = 1024 * 2
    freq, time, Sx = spectrogram(bdot, fs=1e3, nperseg=nfft)
    return Sx, freq, time + t[0]


def plot_ece(shot):
    '''plot density, ece, and b1 psd'''
    ece, t, chs = get_ece(shot)
    Sx, freq, time = get_bdot_spec(shot)
    ne, tne = _data.get_mds('density', shot, tree='efit01')
    nem = savgol_filter(ne, 21, 3)

    t1, t2 = 1.8e3, 2.5e3
    fn = np.vectorize(lambda t: t > t1 and t < t2)
    tind = fn(t)
    tind1 = fn(time)
    tind2 = fn(tne)

    fig, ax = plt.subplots(3, 1, figsize=[8, 8], sharex=True)
    ax[0].plot(tne[tind2], nem[tind2])
    for i in range(np.shape(ece)[0]):
        ax[1].plot(t[tind], ece[i, tind], label=chs[i])
    ax[-1].pcolormesh(time[tind1], freq, np.log10(Sx[:, tind1]), cmap='jet')
    ax[-1].set_ylim(0, 100)
    ax[1].legend(loc=0)
    ax[-1].set_xlabel('time (ms)')
    ax[0].set_ylabel(r'$n_e$')
    ax[1].set_ylabel(r'$T_e$ (keV)')
    ax[-1].set_ylabel(r'$f$ (kHz)')
    ax[0].set_title(f'#{shot}', fontsize=10, x=0.9)
    fig.tight_layout()
    fig.savefig('../fig/ECE.png', dpi=300, transparent=True)


# %%
if __name__ == '__main__':
    shot = 150136
    plot_ece(shot)
