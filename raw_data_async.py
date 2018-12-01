#!/usr/bin/env python3
"""
Download DBS data from MDSplus using multi-threaded method
"""
import time
import numpy as np
from os.path import isfile
import xarray as xr
import concurrent.futures as cof
# %%


def get_dbs(shot: int, trange: '(t1, t2)' = None) -> ('complex array', 'time'):
    hdir = "/cscratch/hongrongjie/LRC/raw_data/"
    filepath = hdir + f"DBS_{shot}.nc"
    ds = xr.open_dataset(filepath)
    tall = ds['time']
    if trange is None:
        time = tall.values
        da, db = [np.empty((8, len(time))) for i in range(2)]
        for i in range(8):
            da[i, :] = ds[f'd{i+1}a'].values
            db[i, :] = ds[f'd{i+1}b'].values
    else:
        t1, t2 = trange
        idx = np.logical_and(t1 <= tall, tall < t2)
        time = ds['time'][idx].values
        da, db = [np.empty((8, len(time))) for i in range(2)]
        for i in range(8):
            da[i, :] = ds[f'd{i+1}a'][idx].values
            db[i, :] = ds[f'd{i+1}b'][idx].values
    return da + 1j * db, time

# %%


def dl_dbs_ch(shot: int, pname: 'pointname') -> ('signal', 'time'):
    import MDSplus
    conn = MDSplus.Connection("atlas.gat.com")
    print(f"Starting {shot}'s {pname}...")
    dat = np.array(conn.get(r"_s=ptdata($,$)", pname, shot))
    del conn
    return dat, pname


def dl_dbs_t(shot: int) -> None:
    conn = MDSplus.Connection('atlas.gat.com')
    dat = np.array(conn.get(f"_s=ptdata2('d1a', {shot})"))
    t = np.array(conn.get("dim_of(_s)"))
    print(f"Shot {shot}'s d1a and time received.")
    del conn
    return dat, t


def dl_dbs_all(shot: int) -> None:

    hdir = "/cscratch/hongrongjie/LRC/raw_data/"
    filepath = f"{hdir}DBS_{shot}.nc"

    if isfile(filepath):
        print(f'{filepath} exists! Pass!')
        return None

    chan = range(1, 9)
    d1a, time = dl_dbs_t(shot)

    pnames = list(f'd{ch}{ph}' for ch in chan for ph in ('a', 'b'))
    pnames.remove('d1a')
    # print(pnames)

    data = xr.Dataset()
    data['d1a'] = (('time'), d1a)
    data.coords['time'] = time
    # print(data)

    with cof.ProcessPoolExecutor() as exe:  # MP preferred
        # with cof.ThreadPoolExecutor(max_workers=8) as exe:  # MT depreciated
        vecs = [exe.submit(dl_dbs_ch, shot, pname) for pname in pnames]
        for i, f in enumerate(cof.as_completed(vecs)):
            value, pname = f.result()
            print(f"Shot {shot}'s {pname} received.")
            data[pname] = (('time'), value)

    data.to_netcdf(filepath)
    print(f"Shot {shot}'s saved!")


if __name__ == '__main__':
    shots = range(150136, 150142)
    for shot in shots:
        t0 = time.time()
        dl_dbs_all(shot)
        print(f"{time.time() - t0:0.2f} sec")
