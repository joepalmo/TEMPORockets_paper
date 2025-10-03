import xarray as xr
import numpy as np
import seaborn as sns
import glob
from datetime import datetime, timedelta
import netCDF4 as nc
import os
from pathlib import Path
import re, datetime
from scipy.optimize import curve_fit

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

from tempo import *

import os, glob
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

TEMPO_DATA_PATH = "/home/jpalmo/fs09/Datasets/TEMPO"

###
# Load launch vehicle database and convert numeric columns
lv_df = pd.read_csv("https://planet4589.org/space/gcat/tsv/tables/lv.tsv", sep="\t")
# Clean numeric columns 
numeric_cols = ['LV_Min_Stage', 'LV_Max_Stage']
lv_df[numeric_cols] = lv_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# Clean values that might be strings representing numbers
for col in ['Length', 'Diameter', 'Launch_Mass', 'LEO_Capacity', 'GTO_Capacity', 'TO_Thrust', 'Apogee', 'Range']:
    lv_df[col] = pd.to_numeric(lv_df[col].replace(['?', '-', ''], np.nan), errors='coerce')
# Drop the comment row that starts with #
lv_df = lv_df[~lv_df['#LV_Name'].str.contains('#', na=False)]

# load detections
detections_df = pd.read_csv(f'plumes/detections.csv',)
detections_df = pd.concat([detections_df], ignore_index=True)
detections_df['timestamp'] = detections_df['Launch_Date'].apply(clean_and_parse_date)
detections_df["obs_time"] = pd.to_datetime(detections_df["obs_time"])
detections_df = detections_df[detections_df['timestamp'] < pd.Timestamp("2025-09-01")]
# remove Black Brant IX LV_Type
detections_df = detections_df[detections_df['LV_Type'] != 'Black Brant IX']
plume_filepaths = glob.glob(f'plumes/*.nc')
plume_filepaths.sort()
# remove any files that contain '20231014' or '20250718 in the filename
plume_filepaths = [fp for fp in plume_filepaths if '20231014' not in fp and '20250718' not in fp]

# timing -- L2 times
L2_obs_times = []
for j, (i, row)in enumerate(detections_df.iterrows()):
    # find matching TEMPO L2 scan
    # L3_obs_time = row['obs_time'].values[0]
    L3_obs_time = detections_df['obs_time'].values[j]
    # L2_filepath = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L2/NO2/TEMPO_NO2_L2_V03_{L3_obs_time.astype("datetime64[s]").astype(datetime.datetime).strftime("%Y%m%dT%H")}*.nc')[0]
    # L3_filepath = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2/TEMPO_NO2_L3_V03_{L3_obs_time.astype("datetime64[s]").astype(datetime.datetime).strftime("%Y%m%dT%H")}*.nc')[0]
    L2_filepath = next_file_after(
        f"{TEMPO_DATA_PATH}/L2/NO2",
        "TEMPO_NO2_L2_V03_",
        L3_obs_time
    )
    L3_filepath = next_file_after(
        f"{TEMPO_DATA_PATH}/L3/NO2",
        "TEMPO_NO2_L3_V03_",
        L3_obs_time
    )
    L2_ds = TEMPO_NO2_loader(L2_filepath)
    
    # get the centroid of the plume
    plume_ds = xr.open_dataset(plume_filepaths[j])
    centroid_lat, centroid_lon = heatmap_centroid(plume_ds)

    # get the nearest pixel observation time from the L2 product
    L2_obs_time = time_from_lat(L2_ds, centroid_lat)

    # print(L3_obs_time, L2_obs_time)

    # get difference in times, in minutes
    time_diff = L2_obs_time - L3_obs_time
    time_diff_minutes = time_diff / np.timedelta64(1, 'm')
    # print(time_diff_minutes)

    L2_obs_times.append(L2_obs_time)

detections_df['L2_obs_time'] = L2_obs_times
detections_df['L2_time_delta'] = detections_df['L2_obs_time'] - detections_df['timestamp']
detections_df['L2_time_delta'] = pd.to_timedelta(detections_df['L2_time_delta'])
# convert to minutes
detections_df['time_delta_min'] = detections_df['L2_time_delta'].astype('timedelta64[ns]').dt.total_seconds() / 60
# Apply the following criteria only for Falcon 9 launches (if LV_Type is not Falcon 9, then keep):
# I want to keep all detections where SNR >= 6
# For points where time_delta_min > 60, I want to keep detections where SNR >= 5.5
detections_df = detections_df[
    ((detections_df['SNR'] >= 6)) |
    ((detections_df['size'] > 100) & (detections_df['SNR'] >= 5.6) & (detections_df['ime']<9000)) |
    ((detections_df['time_delta_min'] >= 60) & (detections_df['SNR'] >= 5.5) & detections_df['LV_Type'].eq('Falcon 9'))
]
detections_df = detections_df[detections_df['size']>10]
# remove false positives
false_pos_2 = pd.Timestamp("2025-03-31")
false_pos_4 = pd.Timestamp("2025-06-10 15")
false_pos_5 = pd.Timestamp("2024-11-11 18")
# Remove rows that occurred on those days
detections_df = detections_df[detections_df["timestamp"].dt.date != false_pos_2.date()]
detections_df = detections_df[detections_df["obs_time"].dt.floor('H') != false_pos_4]
detections_df = detections_df[detections_df["obs_time"].dt.floor('H') != false_pos_5]

# merge with launch vehicle data
# only keep the first detection for each launch (based on Launch_JD)
merged_detections_df = detections_df.drop_duplicates(subset='Launch_JD', keep='first').merge(
    lv_df,
    left_on=['LV_Type', 'Variant'],
    right_on=['#LV_Name', 'LV_Variant'],
    how='left'
)
# only show detections with SNR > 6
merged_detections_df = merged_detections_df[(merged_detections_df['SNR'] > 6) | ((merged_detections_df['size'] > 100) & (merged_detections_df['SNR'] > 5.6))]
merged_detections_df = merged_detections_df[merged_detections_df['ime'] < 9000]
merged_detections_df = merged_detections_df[merged_detections_df['size'] > 10]
initial_detections_df = merged_detections_df.drop_duplicates(subset='Launch_JD', keep='first')
# Calculate scaled NO₂ at t=0, if time_delta_min > 0
tau = 2.31  # lifetime in hours
initial_detections_df['ppf'] = np.exp(initial_detections_df['time_delta_min'] / 60 / tau)
initial_detections_df.loc[initial_detections_df['time_delta_min'] > 0, 'no2_at_t0'] = initial_detections_df['enhancement_molno2'] * initial_detections_df['ppf']
initial_detections_df.loc[initial_detections_df['time_delta_min'] > 0, 'ime_at_t0'] = initial_detections_df['ime'] * initial_detections_df['ppf']
initial_detections_df = initial_detections_df[initial_detections_df['time_delta_min'] > 0]


# save data -- detections_df, initial_detections_df
detections_df.to_csv('data/detections_final.csv', index=False)
initial_detections_df.to_csv('data/initial_detections.csv', index=False)

# in separate notebooks
# generate figures
# generate supplementary figures