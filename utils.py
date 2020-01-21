from netCDF4 import Dataset
from random import shuffle
import pickle
import pandas as pd
import numpy as np
import os, glob, resource


def extract_nc(path, latbounds, lonbounds):
    try:
        with Dataset(path) as rootgrp:
            lats = rootgrp.variables['north_latitude'][:]
            lons = rootgrp.variables['east_longitude'][:]

            lat_idxs = np.where((lats >= latbounds[0]) & (lats <= latbounds[1]))[0]
            lon_idxs = np.where((lons >= lonbounds[0]) & (lons <= lonbounds[1]))[1]

            data = rootgrp.variables["GHI"][:][min(lat_idxs):max(lat_idxs), min(lon_idxs):max(lon_idxs), ]
    except FileNotFoundError:
        data = np.array([])
        print(path)
    return data


def cache_daily_stacks(ghi_log_file, cache_dir, island_name=None, latbounds=None, lonbounds=None):

    catalog = pd.read_parquet(ghi_log_file)
    dat = catalog[['time_stamp', "GHI"]].dropna()
    dat.index = dat['time_stamp']
    dat.index.name = None

    # Localize to Hawaii
    dat = dat.tz_localize('utc')
    dat = dat.tz_convert('US/Hawaii')
    # Filter by for only day times
    dat_filtered = dat.between_time('09:00:00', '17:00:00').sort_values(by='time_stamp')
    dat_by_day = dat_filtered.groupby(pd.Grouper(freq='D'), as_index=False)

    idx = 0
    for name, group in dat_by_day:
        # Any days with less than 1 hour of data, skip
        if len(group) < 4:
            continue
        # Extract GHI data for this day
        day = np.array([extract_nc(row["GHI"], latbounds, lonbounds)
                        for idx, row in group.iterrows()])

        shape = day.shape
        shape_name = os.path.join(cache_dir, island_name + "_day_{:03d}.pkl".format(idx))
        with open(shape_name, "wb") as f:
            pickle.dump(shape, f)

        # Create a memmap for this day
        filename = os.path.join(cache_dir, island_name + "_day_{:03d}.dat".format(idx))
        fp = np.memmap(filename, dtype='float16', mode="w+", shape=shape)
        fp[:] = day[:]
        del fp

        idx += 1

    return island_name


def make_labels(data, shift_factor):
    frames = []
    shifted = []
    for days in data:
        real, advance = shift(days, shift_factor)
        frames.append(real)
        shifted.append(advance)

    return frames, shifted


def shift(arr, num):

    data = arr[:-num]
    shifted = arr[num:]

    return data, shifted


def load_cache(islands, cache_dir):
    # Need to open a balls load of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (5120, 5120))

    file_paths = list()
    shape_paths = list()
    for isle in islands:
        file_paths += sorted(glob.glob("{}/{}*.dat".format(cache_dir, isle)))
        shape_paths += sorted(glob.glob("{}/{}*.pkl".format(cache_dir, isle)))

    shapes = []
    for pickles in shape_paths:
        with open(pickles, "rb") as f:
            shapes.append(pickle.load(f))

    data = []
    for idx, file in enumerate(file_paths):
        ghi = np.memmap(file, dtype='float16', shape=shapes[idx] + (1,))
        data.append(ghi)
    return data


def normalize(data, x_min, x_max):
    # Normalize x' = (x - xmin) / (xmax - xmin)
    return [(arr - x_min) / (x_max - x_min) for arr in data]


def daily_generator(X, Y):
    while True:
        dat = list(zip(X, Y))
        shuffle(dat)

        for x, y in dat:
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            yield x, y

