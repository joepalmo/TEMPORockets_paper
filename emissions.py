import xarray as xr
import numpy as np
import glob
from datetime import timedelta
import netCDF4 as nc
import os
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings("ignore")

from xarray.backends import BackendEntrypoint

from detect_plumes import detect_single_plume, detect_single_plume1
from launches import *
from utils import *

#### Constants
Na = 6.022e23
no2_molar_mass = 46.0055e-3  # kg/mol

# load launches
# tempo_df = get_launch_df()

# # load GEOS-CF
# geoscf_ds = load_geoscf()

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
        raw = xr.open_dataset(filename_or_obj,)
        return raw.latitude.values, raw.longitude.values, raw.time.values

    def cloud_fraction(self, filename_or_obj):
        raw = xr.open_dataset(filename_or_obj, group='support_data')
        return raw['eff_cloud_fraction'].values[0]
    
def pixel_area_cm2(lat, dlat=0.02, dlon=0.02):
    R = 6.371e6  # Earth radius in meters
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    lat_rad = np.deg2rad(lat)

    area_m2 = (R * dlat_rad) * (R * dlon_rad * np.cos(lat_rad))
    return area_m2 * 1e4  # cm²
    
def load_tempo_data(time, lat, lon):
    """
    Load TEMPO data for the given time and location.
    :param time: The time of the emission event.
    :type time: datetime.datetime
    :param lat: The latitude of the emission event.
    :type lat: float
    :param lon: The longitude of the emission event.
    :type lon: float
    """

    # make sure that t has a specific hour and minute, if not then skip
    if time.hour == 0:
        print("no time", flush=True)
    
    # Define the time range: an hour before to 3 hours after
    time_range = range(-1, 4)  # From -1 to +3 hours

    # List to store matching files
    all_files = []

        # Loop through each offset in the time range
    for offset in time_range:
        # Calculate the time with the offset
        offset_time = time + timedelta(hours=offset)
        # Format the time as a string
        hour_str = offset_time.strftime('%Y%m%dT%H')
        # Find files matching the pattern for this hour
        files = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2/TEMPO_NO2_L3_V03*{hour_str}*')
        # Append the found files to the list
        all_files.extend(files)
        # below is the list of files available for that hour. Adapt to get the files for the other hours
        files = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2/TEMPO_NO2_L3_V03*{hour_str}*')

    # Open the dataset
    try:
        ds = xr.open_mfdataset(all_files, group='product', engine=TEMPOBackendEntrypoint)
    except Exception as e:
        print(f"Error opening dataset: {e}", flush=True)
        return

    # clean dataset
    # "good" data only
    try:
        ds = ds.where(ds['main_data_quality_flag'] == 0)
        # cloud filter recommended by TEMPO team
        ds = ds.where(ds['eff_cloud_fraction'] < 0.4)
        # ds = ds.where(ds['eff_cloud_fraction'] < 0.4)
    except Exception as e:
        print(f"Error cleaning dataset: {e}", flush=True)
        return
    
    bounding_box = (lon - 1, lon + 1, lat - 1, lat + 1)
    lon_min, lon_max, lat_min, lat_max = bounding_box

    # limit the dataset to the bounding box, and subtract the 10th percentile to get enhancements
    ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), drop=True)

    return ds

def calculate_emissions(tempo_df, geoscf_ds, time, lat, lon, figs=False, background_pct=0.05, plume_pct=0.9, outpath="/home/jpalmo/fs09/Projects/TEMPO_rocket_launches/plumes"):
    """
    def calculate_emissions(time, lat, lon):
        Calculate emissions based on the given time and location.
        :param time: The time of the emission event.
        :type time: datetime.datetime
        :param lat: The latitude of the emission event.
        :type lat: float
        :param lon: The longitude of the emission event.
        :type lon: float
        :param figs: Whether to save figures.
        :type figs: bool
        :param background_pct: The percentile for background subtraction.
        :type background_pct: float
        :param plume_pct: The percentile for plume detection.
        :type plume_pct: float
        :return: None
    """
    # Load the TEMPO data
    ds = load_tempo_data(time, lat, lon)
    
    # Calculate the emissions
    rows = []
    # first try the data for the hour of the launch, then try the data for the hour after the launch
    for hour_after_launch in range(1,4):
        try:
            data = ds.isel(time=hour_after_launch)['vertical_column_troposphere']
        except:
            print(f"No data for {hour_after_launch} hours after launch", flush=True)
            continue

        data = data.chunk({'latitude': -1, 'longitude': -1})
        # enhancements = data - data.quantile(0.1)
        enhancements = data - data.quantile(background_pct)

        vals_big = enhancements.values
        finite_big = np.isfinite(vals_big)
        global_med = np.nanmedian(vals_big[finite_big])
        global_mad = np.nanmedian(np.abs(vals_big[finite_big] - global_med))
        global_sigma = 1.4826*global_mad if global_mad > 0 else np.nanstd(vals_big[finite_big])

        plume = enhancements.copy(deep=True)

        # more strict bounding box for plume search
        bounding_box = (lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5)
        lon_min, lon_max, lat_min, lat_max = bounding_box
        plume = plume.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), drop=True)

        # check that there are non nan values in plume
        if np.isnan(plume).all():
            print("No data", flush=True)
            continue

        # Detect the plume
        threshold = enhancements.quantile(plume_pct).values
        
        try:
            bounding_box = (lon - 0.25, lon + 0.25, lat - 0.25, lat + 0.25)
            detected_plume = detect_single_plume(plume, bounding_box, (lat, lon), threshold, min_plume_size=10, snr_threshold=5.5, max_busy_frac=0.2,)
            # detected_plume = detect_single_plume1(
            #     plume, bounding_box, (lat, lon), threshold,
            #     min_plume_size=10, snr_threshold=5,
            #     z_thr=2.0, max_busy_frac=0.2,
            #     k_total=6.0, k_mean=1.5, k_peak=3.0,
            #     global_bg_med=global_med, global_bg_sigma=global_sigma
            #     )
        except Exception as e:
            print(f"Error: {e}", flush=True)
            if figs:
                fig, ax = plt.subplots(figsize=(4,3), dpi=200,)
                plume.plot(ax=ax, vmin=-8e15, vmax=8e15, cmap='RdBu_r')
                ax.set_title(f"$\Delta NO2$ \n {time.strftime('%Y-%m-%d %H:%M')} \n {hour_after_launch} hours after launch")
                # save the figure
                # plt.show()
                plt.savefig(f"{outpath}/figs/nondetections/{time.strftime('%Y%m%dT%H')}_hour{hour_after_launch}_noplume.png")
                plt.close()
            continue

        isolated_plume = plume * detected_plume
        isolated_plume.attrs = detected_plume.attrs

        L = L_geoscf(geoscf_ds, time, lat, lon)

        # Compute pixel area (1D over lat)
        plume_pixel_area_1d = pixel_area_cm2(isolated_plume.latitude)
        # Broadcast to 2D
        plume_pixel_area_2d = xr.broadcast(plume_pixel_area_1d, isolated_plume)[0]
        # Calculate ime
        plume_mass = isolated_plume * (no2_molar_mass / Na) # convert to kg NO2 / cm2
        nox_plume_mass = (plume_mass*L) # convert to kg NOx / cm2
        ime = (nox_plume_mass * plume_pixel_area_2d).sum().values # convert to kg NOx

        # calculate mol of NO2 in plume
        mol_no2 = (isolated_plume * plume_pixel_area_2d).sum().values / Na # convert to mol NO2


        # only keep plumes with ime > 200 kg
        if ime < 200:
            print("False positive, plume too small", flush=True)
            continue

        #only keep plumes where more than 75% of the pixels are non nan
        if np.isnan(isolated_plume).sum() / isolated_plume.size > 0.25:
            print("False positive, too many nans", flush=True)
            continue

        # save the plume to a netcdf file
        # create output directory if it doesn't exist
        os.makedirs(outpath, exist_ok=True)
        isolated_plume.to_netcdf(f"{outpath}/{time.strftime('%Y%m%dT%H')}_hour{hour_after_launch}.nc")

        if figs:
            fig, axs = plt.subplots(1,2, figsize=(8,3), dpi=100, sharex=True, sharey=True)
            im0 = plume.plot(ax=axs[0], vmin=-8e15, vmax=8e15, cmap='RdBu_r')
            detected_plume.plot(ax=axs[1],)
            im0.colorbar.set_label(r'$\Delta$NO$_2$ [molec/cm$^2$]')
            axs[0].set_title(f"$\Delta NO2$ \n {time}")
            axs[1].set_title(f"Detected plume \n NOx mass = {ime:.2f} kg")
            axs[1].set_axis_off();
            fig.tight_layout();
            # save the figure
            # plt.show()
            plt.savefig(f"{outpath}/figs/detections/{time.strftime('%Y%m%dT%H')}_hour{hour_after_launch}.png")
            plt.close()

        # save ime to dataframe
        row = tempo_df.loc[time]
        row['L'] = L
        row['ime'] = ime
        row['SNR'] = detected_plume.attrs['snr']
        row['distance'] = detected_plume.attrs['distance']
        row['enhancement'] = detected_plume.attrs['enhancement']
        row['enhancement_molno2'] = mol_no2
        row['mean_signal'] = detected_plume.attrs['mean_signal']
        row['size'] = detected_plume.attrs['size']
        row['hour_after_launch'] = hour_after_launch
        row['obs_time'] = data.time.values
        row['time_delta'] = data.time.values - time
        row['time_delta'] = pd.to_timedelta(row['time_delta'])

        rows.append(row)
    
    if len(rows)>0:
        return pd.DataFrame(rows)
    else:
        print("No plume detected", flush=True)
        return

