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
from numpy.core.multiarray import ndarray
from scipy.signal import welch, correlate, spectrogram, \
    coherence, savgol_filter as sgf
from raw_data_async import get_dbs
import os.path

# import _axes_prop
# plt.switch_backend('Agg')
sns.set(style='ticks', palette='Set2')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})


# %%
def freq_filter(spec: '2D array of each quadrature spectra',
                freq: '1D array',
                del_f: 'float' = 500) -> 'Boolean index':
    """Determine peak of spectra and apply a window with del_f = 500kHz"""
    spec_f = np.mean(spec, axis=-1)  # average over time
    spec_m = sgf(spec_f, 41, 3)
    ipk = np.argmax(spec_m)
    fpk = freq[ipk]
    print(f"f_peak = {fpk} kHz")
    fidx = np.logical_and.reduce((freq > fpk - del_f, freq < fpk + del_f))
    return fidx


def calc_ne_rms_coh(data: 'dbs array', t: 'dbs time',
                    nperseg: int = 1024, nfft: int = 1024 * 4,
                    overlap=0.95) -> ('ne', 'time'):
    """Calculate ne rms level by integrating spectrogram along freq axis"""
    da = data.real
    db = data.imag
    noverlap = int(nperseg * overlap)
    freq, time, spec_a = spectrogram(da, nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap, fs=5e3)
    freq, coh = coherence(da, db, nperseg=nperseg, nfft=nfft,
                          noverlap=noverlap, fs=5e3)
    ind = freq > 100
    spec = spec_a * coh[:, :, None]
    ne = np.empty((8, len(time)))
    for i in range(8):
        if i == 3:
            ind = freq > 800
        spec_m = spec[i, ind, :]
        fidx = freq_filter(spec_m, freq[ind])
        ne[i, :] = np.sum(spec_m[fidx, :], axis=0)
    return ne, time + t[0]


def calc_ne_rms_quad(data: 'dbs array', t: 'dbs time',
                     nperseg: int = 1024, nfft: int = 1024 * 2,
                     overlap=0.9) -> ('ne', 'time'):
    """Calculate ne rms level by integrating spectrogram along freq axis"""
    noverlap = int(nperseg * overlap)
    freq, time, spec = spectrogram(data, nperseg=nperseg, nfft=nfft,
                                   noverlap=noverlap, fs=5e3,
                                   return_onesided=False)
    ne = np.empty((8, len(time)))
    for i in range(8):
        fidx = freq_filter(spec[i, :, :], freq)
        ne[i, :] = np.sum(spec[i, fidx, :], axis=0)
    return ne, time + t[0]


def calc_coh_all(ne: 'density array', nperseg: int = 1024 * 2,
                 nfft: int = 1024 * 2, fs=5e3, overlap=0.9,
                 iref: 'ref channel' = 2) -> ('coherence', 'freq: array'):
    """Calculate coherence between each channels"""
    nem = (ne - np.mean(ne, axis=-1, keepdims=True)) / np.std(ne, axis=-1,
                                                              keepdims=True)
    xcoh = np.empty((nem.shape[0], nfft // 2 + 1))
    noverlap = int(nperseg * overlap)
    freq: ndarray = np.fft.fftfreq(nfft, 1 / fs)
    for i in range(ne.shape[0]):
        _, xcoh[i, :] = coherence(nem[iref, :], nem[i, :], fs=fs,
                                  nperseg=nperseg, nfft=nfft,
                                  noverlap=noverlap)
    return xcoh, freq


def calc_hurst(ne1: 'nd array of density rms'):
    """Calcualte Hurst exponent of each channel"""
    from Hurst import compute_Hc
    for i in range(ne1.shape[0]):
        y = ne1[i, :]
        hurst, c, data = compute_Hc(y, n=2, kind='price', simplified=True)
        fig, ax = plt.subplots()
        plt.plot(data[0], c * data[0] ** hurst)
        plt.scatter(data[0], data[1])
        ax.set(yscale='log', xscale='log', ylabel='R/S ratio',
               xlabel='Time interval', title=f"ch{i + 1} H={hurst:.4f}")
        plt.tight_layout()
        plt.savefig(f'../fig/Hurst_ch{i + 1}.pdf')
        plt.close(fig)


def calc_autospec(ne1, nfft=2048):
    f1 = np.linspace(0.1, 50)
    for i in range(ne.shape[0]):
        y = ne1[i, :]
        freq, pwr = welch(y, fs=fs1, nperseg=nfft, noverlap=int(0.95 * nfft))
        fig, ax = plt.subplots(figsize=[5, 4])
        plt.loglog(freq, pwr / pwr.max(), lw=0.8)
        plt.loglog(f1, f1 ** -1 / 10, c='C1')
        plt.text(20, 0.01, r'$f^{-1}$', color='C1', fontsize=14)
        ax.set(xlabel='f (kHz)', ylabel='auto-spectra', ylim=[1e-6, 1],
               title=f'Channel {i + 1}')
        plt.tight_layout()
        plt.savefig(f'../fig/auto-spec_ch{i + 1}.pdf')
        plt.show()


@nba.jit()
def calc_ccf(im: 'nd-array of dbs signals', t: '1d array',
             iref=2) -> ('cross correlation', 'time lag'):
    """Calculate CCF of multi-channel signals"""
    tlen = len(t)
    dn = im.shape[0]
    xcorr = np.empty((dn, tlen * 2 - 1))
    for ir in range(dn):
        x, y = im[iref, :], im[ir, :]
        x -= x.mean()
        y -= y.mean()
        corr = correlate(x, y, mode='full', method='auto')
        corr /= (np.std(im[iref, :]) * np.std(im[ir, :]) * tlen)
        xcorr[ir, :] = corr
    dt = np.mean(np.diff(t))
    tlag = np.arange(-tlen + 1, tlen) * dt
    return xcorr, tlag


def plot_ccf(corr: 'cross correlation', tlag: 'time lag',
             iref=2) -> None:
    fig, ax = plt.subplots(figsize=[6, 5])
    for ch in range(corr.shape[0]):
        if ch != iref:
            tmax = tlag[np.argmax(np.abs(corr[ch, :]))] * 1e3
            plt.plot(tlag, corr[ch, :],
                     label=f'ch{iref + 1}-{ch + 1}: {tmax:.2f} us')
    plt.axvline(0, c='gray', ls='--')
    plt.axhline(0, c='gray', ls='--')
    plt.legend(loc=1, fontsize=9)
    ax.set(xlabel='time lag (ms)', ylabel='cross correlation',
           xlim=[-.5, .5], ylim=[-1, 1])
    plt.tight_layout()
    plt.savefig(f'../fig/ccf_rms_1d.pdf', transparent=True)
    plt.show()


def plot_ne_time(ne, time) -> None:
    """Plot ne fluctuation time series"""
    fig, axs = plt.subplots(4, 2, sharex='col', figsize=[10, 6])
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.plot(time, ne[i, :], '-', lw=0.8)
        ax.set_ylabel(f'n{i + 1}')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig('../fig/ne_rms_fft_time.pdf', transparent=True)
    plt.show()


def plot_ne_coh_all(xcoh, freq, iref=2) -> None:
    """Plot coherence between different channels."""
    plt.figure()
    for i in range(xcoh.shape[0]):
        if i != iref:
            plt.plot(freq, xcoh[i, :], lw=1, label=f'ch{iref + 1}-{i + 1}')
    plt.legend(loc=0)
    plt.xlabel('f (kHz)')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('../fig/coherence_rms_fft.pdf', transparent=True)
    plt.show()


def plot_ne_coh_pair(xcoh, freq,
                     chans: ('channel 1', 'channel 2')) -> None:
    ch1, ch2 = chans
    plt.figure()
    plt.semilogx(freq, xcoh, '-', lw=0.8, label=f'ch{ch1}-{ch2}')
    plt.legend(loc=0)
    plt.xlabel('f (kHz)')
    plt.ylabel('Coherence')
    # plt.xlim(1, 000)
    plt.ylim(0, 1)
    plt.savefig(f'../fig/coherence_ch{ch1}-{ch2}.pdf')
    plt.show()


# %%
if __name__ == "__main__":
    shot = 150136
    fs = 5e3
    iref = 0
    t1, t2 = 2500, 2600
    nfft0: int = 1024
    readne = True

    filepath = f'../proc_data/fft_ne_{shot}_{t1}-{t2}_fpeak_coh.nc'
    if os.path.isfile(filepath) and readne:
        ds = xr.open_dataset(filepath)
        ne = ds['ne'].values
        time = ds['time'].values
        print("Loaded processed nc file.")
    else:
        data, t = get_dbs(shot, (t1, t2))
        print("Read DBS raw data.")
        ne, time = calc_ne_rms_coh(data, t, nperseg=nfft0, nfft=nfft0 * 2,
                                   overlap=0.95)
        print("Start saving ne data.")
        chan = np.arange(1, 9)
        ds = xr.Dataset({'ne': (('chan', 'time'), ne)},
                        coords={'chan': chan, 'time': time})
        ds.to_netcdf(filepath)
        print("Saved processed data.")

    # %%
    t3, t4 = 2572, 2575
    ece, tece = _data.get_mds('ece5', shot)
    ece1 = np.interp(time, tece, ece)
    # ind = ece1 <= 0.79
    ind = np.logical_and.reduce((time > t3, time < t4, ece1 < 0.8))
    ne1, time1 = ne[:, ind], time[ind]
    fs1 = np.reciprocal(np.min(np.diff(time1)))  # new sampling rate
    print(f"ne1 shape = {ne1.shape}")

    # %%
    # plot time series
    from scipy.interpolate import interp1d

    iref = 0
    num = 10000

    time2 = np.linspace(time1.min(), time1.max(), num=num)
    ne2 = np.empty((8, num))
    for i in range(ne1.shape[0]):
        foo = interp1d(time1, ne1[i, :], kind='cubic')
        ne2[i, :] = foo(time2)

    print(f"ne2 shape {ne2.shape}")
    plot_ne_time(ne1, time1)
    # %%
    # # Calculate CCF

    xcorr, tlag = calc_ccf(ne2, time2, iref=iref)

    tlag1 = np.linspace(-2, 2, num=num)
    xcorr1 = np.empty((8, num))
    for i in range(xcorr.shape[0]):
        foo = interp1d(tlag, xcorr[i, :], kind='cubic')
        xcorr1[i, :] = foo(tlag1)

    plot_ccf(xcorr, tlag, iref=iref)
    plot_ccf(xcorr1, tlag1, iref=iref)

    # %%
    # Calculate Hurst exponent
    # calc_hurst(ne1)
    # %%
    # # Plot auto-spectra
    # calc_autospec(ne1, nfft=2048)
    # %%
    # # Plot coherence
    # plot_ne_time(ne1, time1)
    # nfft = 1024 * 1
    # # plotall = True
    # plotall = False
    # if plotall:
    #     xcoh, freq = calc_coh_all(ne1, nfft=nfft, fs=fs1, iref=iref)
    #     plot_ne_time(ne1, time1)
    #     plot_ne_coh_all(xcoh, freq, iref=iref)
    # else:
    #     ch1 = 4
    #     x = ne1[ch1 - 1]
    #     x = (x - x.mean()) / x.std()
    #
    #     for ch2 in range(1, 9):
    #         if ch2 is not ch1:
    #             y = ne1[ch2 - 1, :]
    #         y = (y - y.mean()) / y.std()
    #         freq, xcoh = coherence(x, y, fs=fs1, nperseg=nfft,
    #                                nfft=nfft * 2, noverlap=int(0.8 * nfft))
    #         plot_ne_coh_pair(xcoh, freq, (ch1, ch2))
