#!/usr/bin/env python3
"""
Libaray for download and read raw DBS data
Depreciated due to Sequential routine
"""
# from itertools import product
from os.path import isfile
import xarray as xr
# import _data
import numpy as np


# %%
def dl_dbs(shot):
    """download each DBS channel (in-phase and quadrature separately) """
    import MDSplus
    con = MDSplus.Connection("atlas.gat.com")
    chan = np.arange(1, 9)

    da, db = np.array([]), np.array([])
    # Real part
    for ch in chan:
        pname = f'd{ch}a'
        print(f"Downloading {pname} for {shot}...")
        da1 = con.get("_s=ptdata2($,$)", pname, shot)
        da = np.append(da, da1)

    # Imaginary part
    for ch in chan:
        pname = f'd{ch}b'
        print(f"Downloading {pname} for {shot}...")
        db1 = con.get("_s=ptdata2($,$)", pname, shot)
        db = np.append(db, db1)

    t = np.array(con.get("dim_of(_s)"))

    da = np.reshape(da, (8, len(t)))
    db = np.reshape(db, (8, len(t)))
    return da, db, t


def save_dbs(shot):
    """Save DBS data to nc file"""
    if isfile(filepath):
        print(f"File for {shot} exist!")
    else:
        ya, yb, t = dl_dbs(shot)
        chan = np.arange(1, 9)

        comp = dict(zlib=True, complevel=5)
        dat = xr.Dataset({'real': (['chan', 'time'], ya),
                          'imag': (['chan', 'time'], yb)},
                         coords={'chan': chan, 'time': t})
        encoding = {var: comp for var in dat.data_vars}
        print(f"Saving to {filepath}")
        dat.to_netcdf(filepath, encoding=encoding)
        print(f"Done with {shot}!")


def get_dbs(shot, trange=None):
    """
    Load DBS of each channel
    trange is a two-variable tuple or list for time range slice
    """
    dat = xr.open_dataset(filepath)
    tall = dat['time']
    if trange is None:
        time = tall
        da, db = dat['real'], dat[f'imag']
    else:
        t1, t2 = trange
        idx = (tall > t1) & (tall < t2)
        time = dat['time'][idx]
        da, db = dat['real'][:, idx], dat[f'imag'][:, idx]
    return da.values + 1j * db.values, time.values


# %%
if __name__ == "__main__":

    # shot = 150136
    shots = range(150136, 150142)
    for shot in shots:
        filepath = f'../raw_data/DBS_{shot}.nc'
        save_dbs(shot)
