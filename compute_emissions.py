import xarray as xr
import numpy as np
import glob
from datetime import datetime, timedelta
import netCDF4 as nc
import os

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings("ignore")

from xarray.backends import BackendEntrypoint

from detect_plumes import detect_single_plume
from launches import *
from utils import *
from emissions import *

# make a directory for the current date
outdir = f'plumes'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    os.makedirs(outdir+'/figs/')
    os.makedirs(outdir+'/figs/detections/')
    os.makedirs(outdir+'/figs/nondetections/')

# load launches
tempo_df = get_launch_df()
tempo_df = tempo_df.loc['2023-08-01':]

# load GEOS-CF
geoscf_ds = load_geoscf()

nondetections = []
detections = []
# loop through all the launches
for i, (time, row) in enumerate(tempo_df.iterrows()):
    print(f"Processing {time}")

    lat = row['lat']
    lon = row['lon']

    d = calculate_emissions(tempo_df, geoscf_ds, time, lat, lon, figs=True, outpath=outdir,)
    
    if d is not None and not d.empty:
        detections.append(d)

# Combine all DataFrames into one
detections_df = pd.concat(detections, ignore_index=True)

# Save
detections_df.to_csv(f'{outdir}/detections.csv', index=False)
