import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import dask
import scipy
import glob
from tqdm import tqdm
from datetime import datetime
import pytz

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.img_tiles as cimgt
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import imageio

from xarray.backends import BackendEntrypoint

# to create the .mp4 files for the false positives:
# python make_animation.py --obs "NO2" --start "2025-03-29" --end "2025-04-01" --bounding_box "(-82.14,27.7,-79.35,29.2)" --output "/home/jpalmo/fs09/Projects/TEMPO_rocket_launches/paper_final/false_positives/20250331"
# python make_animation.py --obs "NO2" --start "2025-06-09" --end "2025-06-11" --bounding_box "(-82.14,27.7,-79.35,29.2)" --output "/home/jpalmo/fs09/Projects/TEMPO_rocket_launches/paper_final/false_positives/20250610"
# python make_animation.py --obs "NO2" --start "2024-11-10" --end "2024-11-12" --bounding_box "(-82.14,27.7,-79.35,29.2)" --output "/home/jpalmo/fs09/Projects/TEMPO_rocket_launches/paper_final/false_positives/20241111"


### PARSE ARGS

parser = argparse.ArgumentParser(
                    prog='TEMPO animation',
                    description='Process TEMPO files and make an animation',)
parser.add_argument('--obs', type=str, default='NO2', choices=['NO2', 'HCHO', 'O3', 'NO2_strat'], help='type of observation')
parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
parser.add_argument('--bounding_box', default="(-134,20,-45,60)", help='Bounding box')
parser.add_argument('--output', type=str, help='Output directory')

args = parser.parse_args()
args.bounding_box = eval(args.bounding_box)

class TEMPOBackendEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs):
        ds = xr.open_dataset(filename_or_obj, group='product',)
        lat, lon, time = self.lat_lon_time(filename_or_obj)
        # Assign new coordinates to the dataset
        ds = ds.assign_coords(latitude=('latitude', lat), longitude=('longitude', lon), time=('time', time))
        try:
            ds['eff_cloud_fraction'] = (['latitude', 'longitude'], self.cloud_fraction(filename_or_obj))
        except:
            pass
        return ds
    
    def lat_lon_time(self, filename_or_obj):
        raw = xr.open_dataset(filename_or_obj)
        return raw.latitude.values, raw.longitude.values, raw.time.values

    def cloud_fraction(self, filename_or_obj):
        raw = xr.open_dataset(filename_or_obj, group='support_data')
        return raw['eff_cloud_fraction'].values[0]

retrieval_dict = {'NO2': 'vertical_column_troposphere', 'HCHO': 'vertical_column', 'O3': 'column_amount_o3', 'NO2_strat': 'vertical_column_stratosphere',}
vmin_dict = {'NO2': 1e15, 'HCHO': 0, 'O3': 200, 'NO2_strat': 1.95e15}
vmax_dict = {'NO2': 7e15, 'HCHO': 4e16, 'O3': 400, 'NO2_strat': 2.25e15}
# cmap_dict = {'NO2': 'Reds', 'HCHO': 'Greens', 'O3': 'Blues', 'NO2_strat': 'Reds'}
cmap_dict = {'NO2': 'plasma', 'HCHO': 'Greens', 'O3': 'Blues', 'NO2_strat': 'Reds'}
cbarlabel_dict = {'NO2': r'$NO_2$ vertical column density $[\frac{molecules}{cm^2}]$', 'HCHO': r'$HCHO$ vertical column density $[\frac{molecules}{cm^2}]$', 'O3': r'Total Column $O_3$ $[DU]$', 'NO2_strat': r'$NO_2$ stratospheric vertical column density $[\frac{molecules}{cm^2}]$'}

if __name__ == "__main__":
    # get files
    # Generate a list of date strings in YYYYMM format
    date_range = pd.date_range(start=args.start, end=args.end, freq='D').strftime("%Y%m%d").tolist()

    # Create a glob pattern
    # glob_patterns = [f"/home/jpalmo/fs09/Projects/TEMPO_apriori/data/raw/L3/{args.obs}/*{date}*.nc" for date in date_range]
    # glob_patterns = [f"/home/jpalmo/fs09/Projects/TEMPO_apriori/data/raw/L3/NO2/*{date}*.nc" for date in date_range]
    glob_patterns = [f"/home/jpalmo/fs09/Datasets/TEMPO/L3/{args.obs}/*{date}*.nc" for date in date_range]

    # List all matching files
    files = []
    for pattern in glob_patterns:
        files.extend(glob.glob(pattern))

    files.sort()
    print(len(files)," files to be opened", flush=True)
    # files = glob.glob(args.path)

    # open dataset
    ds = xr.open_mfdataset(files, group='product', combine='nested', concat_dim='time', engine=TEMPOBackendEntrypoint)
    
    print("Data opened", flush=True)

    # clean dataset
    # "good" data only
    try:
        ds = ds.where(ds['main_data_quality_flag'] == 0)
        # cloud filter recommended by TEMPO team
        # ds = ds.where(ds['eff_cloud_fraction'] < 0.25)
        ds = ds.where(ds['eff_cloud_fraction'] < 0.4)
    except:
        pass

    ds = ds[retrieval_dict[args.obs]]

    # Subset the dataset based on latitude and longitude
    lon_min, lat_min, lon_max, lat_max = args.bounding_box
    ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    ds = ds.sortby('time')
    
    # convert time to Eastern local time
    time=ds['time'].to_index()
    time_utc = time.tz_localize(pytz.UTC)
    time_local = time_utc.tz_convert(pytz.timezone('US/Eastern'))
    time_local = time_local.tz_localize(None)
    ds['time'] = pd.DatetimeIndex(time_local)

    print("Data processed", flush=True)

    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(f'{args.output}/frames'):
        os.makedirs(f'{args.output}/frames')

    # Plotting
    figure, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)

    make_cbar = True
    for i,t in enumerate(ds.time):
        # Define Natural Earth features
        countries = cfeature.BORDERS.with_scale('50m')
        states = cfeature.STATES.with_scale('50m')

        # Set map background and features
        ax.add_feature(countries, edgecolor='black', linewidth=1)
        ax.add_feature(states, edgecolor='black', linewidth=0.25)

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        hour = str(ds.time.values[i])

        if make_cbar:
            p = ds.isel(time=i).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_dict[args.obs], vmin=vmin_dict[args.obs], vmax=vmax_dict[args.obs], add_colorbar=False)
            ax.set_title('')
            cbar_kwargs={'orientation': 'horizontal', 'pad': 0.05}
            cbar = figure.colorbar(p, ax=ax, **cbar_kwargs)
            cbar.ax.tick_params(labelsize=6)  # Set colorbar label size
            cbar.set_label(cbarlabel_dict[args.obs], fontsize=8)  # Set colorbar label fontsize
            make_cbar = False
        else:
            ds.isel(time=i).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_dict[args.obs], vmin=vmin_dict[args.obs], vmax=vmax_dict[args.obs], add_colorbar=False)
            ax.set_title('')

        # Extract datetime and round to the nearest minute
        timestamp = pd.to_datetime(ds.time.values[i]).round('min')
        # Set a clean title manually
        ax.set_title(timestamp.strftime('%Y-%m-%d %H:%M'))

        # add a star Cape Canaveral
        ax.scatter([-80.6819], [28.5233], marker='*', edgecolor='black', color='grey', linewidth=0.5, s=50)
        # add a star Starbase
        # ax.scatter([-97.186389], [25.9875], marker='*', edgecolor='black', color='grey', linewidth=0.5, s=50)

        # plt.title(f'{hour}')
        plt.savefig(f'{args.output}/frames/{hour}.png')
        # Clear the axis
        ax.cla()
        print(f'Image {i} saved', flush=True)

    # create animation
    # choose the right frames
filenames = glob.glob(f'{args.output}/frames/*.png')
filenames.sort()
# Create the GIF
imageio.mimsave(f'{args.output}/output.gif', [imageio.imread(frame) for frame in filenames], duration=400, loop=0)
print(f'GIF saved', flush=True)
# Create the MP4
imageio.mimsave(
    f'{args.output}/output.mp4',
    [imageio.imread(frame) for frame in filenames],
    fps=1  # 1 frame per second = slow playback
)
print(f'MP4 saved', flush=True)

        