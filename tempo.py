import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import dask
import scipy
import glob
from tqdm import tqdm
import datetime
from shapely.geometry import Point, Polygon
import concurrent.futures

import warnings
warnings.filterwarnings("ignore")

import argparse
import os

### UTILITY FUNCTIONS

def convert_seconds_to_datetime(seconds):
    base_time = datetime.datetime(1980, 1, 6, 0, 0, 0)
    delta = datetime.timedelta(seconds=seconds)
    datetime_obj = base_time + delta
    return datetime_obj

def point_in_polygon(lat, lon, geospatial_bounds):
    point = Point(lat, lon)
    polygon = Polygon(geospatial_bounds)
    return polygon.contains(point)

Na = 6.02214076e23
R = 8.31446261815324

def convert_volume_mixing_ratio_to_number_density(pressure, temperature, vmr):
    number_density = (pressure * vmr) / (R * temperature)
    return number_density

# Define a function to save the dataset to netCDF
def save_dataset(d, file_path):
    encoding = {var: {"zlib": True, "complevel": 5} for var in d.data_vars}  # Compression level 5
    d.to_netcdf(file_path, format="NETCDF4", encoding=encoding)


# define a function to load in TEMPO L2 data
def TEMPO_NO2_loader(file_path):
    # Open with netcdf4
    file_id = Dataset(file_path)

    # Extract the latitude and longitude
    lat_2d = file_id['geolocation']['latitude'][:].filled(np.nan)
    lon_2d = file_id['geolocation']['longitude'][:].filled(np.nan)

    geolocation_ds = xr.open_dataset(file_path, group='geolocation')

    chunk_x, chunk_y = lat_2d.shape

    # extract the bounds 
    corner = np.arange(4)
    lat_bounds = file_id['geolocation']['latitude_bounds'][:].filled(np.nan)
    lon_bounds = file_id['geolocation']['longitude_bounds'][:].filled(np.nan)

    # Extract the time
    time = np.vectorize(convert_seconds_to_datetime)(file_id['geolocation']['time'][:].data)
    time_start = file_id.time_coverage_start
    time_end = file_id.time_coverage_end

    # build the dataset
    ds = xr.Dataset(
        {
            "lat_bounds": (["lat", "lon", "corner"], lat_bounds),  # Assign data with lat/lon as dimensions
            "lon_bounds": (["lat", "lon", "corner"], lon_bounds),  # Assign data with lat/lon as dimensions
        },
        coords={
            "time": (["time"], time),  # Assign time as a 1D coordinate
            "lat": (["lat", "lon"], lat_2d),  # Assign lat as a 2D coordinate
            "lon": (["lat", "lon"], lon_2d),  # Assign lon as a 2D coordinate
            "corner": (["corner"], corner)  # Assign pressure levels as a 1D coordinate
        },
        attrs={
            "time_coverage_start": time_start,
            "time_coverage_end": time_end,
        }
    )

    return ds