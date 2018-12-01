#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Get ECE cross correlation contour plots
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
import _data, _misc
# plt.switch_backend('Agg')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4]})
sns.set(style='ticks', palette='Set2')


# %%
def get_ece_h(shot):
    '''get ece data'''
    ece = np.array([])
    ece_h = np.array([])
    chs = []
    fs = 5e3
    b, a = butter(4, [10/fs, 100/fs], btype='bandpass')
    for ch in range(1, 33):
        pname = f'ece{ch}'
        chs.append(pname)
        y, t = _data.get_mds(pname, shot)
        y_h = filtfilt(b, a, y)
        if ece.size == 0:
            ece, ece_h = y, y_h
        else:
            ece = np.vstack((ece, y))
            ece_h = np.vstack((ece_h, y_h))
    return ece, ece_h, t, chs


def calc_ccf(ece, t):
    '''Calculate CCF of ECE signals'''
    fn = _misc.slcfun(t1, t2)
    tind = fn(t)
    im = ece[:, tind]
    dn, tlen = im.shape
    xcorr = np.empty((dn, tlen*2-1))
    iref = 25
    for ir in range(dn):
        corr = correlate(im[iref, :], im[ir, :],
                         mode='full', method='auto')
        corr /= (np.std(im[iref, :]) * np.std(im[ir, :]) * tlen)
        xcorr[ir, :] = corr
    dt = np.mean(np.diff(t))
    tlag = np.arange(-tlen+1, tlen) * dt
    return xcorr, tlag


def plot_ccf(corr, tlag):
    chan = np.arange(1, 33)
    plt.figure()
    plt.pcolormesh(tlag, chan, corr, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel(r'$\Delta t$ (ms)')
    plt.ylabel('channels')
    plt.title(f'#{shot}')
    plt.tight_layout()
    plt.savefig(f'../fig/ECE_CCF.png', dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    shot = 150136
    t1, t2 = 2.3e3, 2.6e3
    ece, eceh, t, chs = get_ece_h(shot)
    corr, tlag = calc_ccf(eceh, t)
    plot_ccf(corr, tlag)
