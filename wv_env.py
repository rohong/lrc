#!/usr/bin/env python3
"""
Wavelet envelope coherence and ccf
Author: Rongjie Hong
Date: 2018-10-26
"""
import numba, numpy as np, matplotlib.pyplot as plt, xarray as xr, seaborn as sns
from waveletFunctions import cwt

sns.set(style='ticks', palette='Set2')
plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})


# %%
def get_Swv_ch(shot, ch):
    """Read wavelet power from local data"""
    filepath = f'../proc_data/wv_{shot}_ch{ch}_t2500-2600.nc'
    dat = xr.open_dataset('../proc_data/wv_{shot}')
    return dat['freq'], dat['time'], dat['Swv']


@numba.jit()
def calc_ne_rms(power, freq):
    """Calculate ne rms level by integrating along frequency axes"""
    ind = (freq > 200) & (freq < 15e2)
