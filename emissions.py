import xarray as xr
import numpy as np
import glob
from datetime import timedelta, datetime
import netCDF4 as nc
import os
import pandas as pd
import cartopy.crs as ccrs
import re
import earthaccess

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
from tempo import *

#### Constants
Na = 6.022e23
no2_molar_mass = 46.0055e-3  # kg/mol

# load launches
# tempo_df = get_launch_df()

# # load GEOS-CF
# geoscf_ds = load_geoscf()

def convert_seconds_to_datetime(seconds):
    base_time = datetime.datetime(1980, 1, 6, 0, 0, 0)
    delta = datetime.timedelta(seconds=seconds)
    datetime_obj = base_time + delta
    return datetime_obj

def mask_and_trim_to_roi(ds, bounding_box):
    lon_min, lon_max, lat_min, lat_max = bounding_box

    # if lat/lon are 1D, mask out values outside the bounding box and trim to the bounding box
    if len(ds.latitude.shape) == 1 and len(ds.longitude.shape) == 1:
        ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), drop=True)

    else:
        roi_mask = (
            ((ds["longitude"] >= lon_min) & (ds["longitude"] <= lon_max)) &
            ((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max))
        )

        # trim to the smallest y/x box that contains the ROI
        # m = roi_mask.values
        m = roi_mask.any(dim="time").values if "time" in roi_mask.dims else roi_mask.values
        ys = np.where(m.any(axis=1))[0]
        xs = np.where(m.any(axis=0))[0]

        if ys.size == 0 or xs.size == 0:
            raise ValueError(f"ROI mask empty: bbox={bounding_box} (no pixels in bounds)")

        ds = ds.isel(y=slice(ys.min(), ys.max()+1), x=slice(xs.min(), xs.max()+1))
    
    return ds

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
    
class TEMPOBackendEntrypoint_L2(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs):
        with nc.Dataset(filename_or_obj) as file_id:
            geo = file_id["geolocation"]

            lat_2d = geo["latitude"][:].filled(np.nan)
            lon_2d = geo["longitude"][:].filled(np.nan)

            lat_bounds = geo["latitude_bounds"][:].filled(np.nan)
            lon_bounds = geo["longitude_bounds"][:].filled(np.nan)

            # time can be 1D or 2D depending on product; keep it truthful
            time_raw = geo["time"][:].filled(np.nan)
            time_dt = np.vectorize(convert_seconds_to_datetime)(time_raw)

            time_start = getattr(file_id, "time_coverage_start", None)
            time_end   = getattr(file_id, "time_coverage_end", None)
            t0 = np.datetime64(time_start)

            # get actual column data
            main_data_quality_flag = file_id["product"]["main_data_quality_flag"][:].filled(0)
            vcd_trop = file_id["product"]["vertical_column_troposphere"][:].filled(np.nan)
            vcd_strat = file_id["product"]["vertical_column_stratosphere"][:].filled(np.nan)
            vcd_trop_uncertainty = file_id["product"]["vertical_column_troposphere_uncertainty"][:].filled(np.nan)

            # cloud fraction
            cloud_fraction = file_id["support_data"]["eff_cloud_fraction"][:].filled(np.nan)

        ny, nx = lat_2d.shape
        corner = np.arange(4)

        ds = xr.Dataset(
            data_vars={
                "lat_bounds": (("y", "x", "corner"), lat_bounds),
                "lon_bounds": (("y", "x", "corner"), lon_bounds),
                "main_data_quality_flag": (("y", "x"), main_data_quality_flag),
                "vertical_column_troposphere": (("y", "x"), vcd_trop),
                "vertical_column_stratosphere": (("y", "x"), vcd_strat),
                "vertical_column_troposphere_uncertainty": (("y", "x"), vcd_trop_uncertainty),
                "eff_cloud_fraction": (("y", "x"), cloud_fraction),
            },
            coords={
                "y": np.arange(ny),
                "x": np.arange(nx),
                "latitude": (("y", "x"), lat_2d),
                "longitude": (("y", "x"), lon_2d),
                "corner": corner,
                "time": t0,
            },
            attrs={
                "time_coverage_start": time_start,
                "time_coverage_end": time_end,
            },
        )

        # attach time in a way that matches its real shape
        if time_dt.ndim == 1:
            ds = ds.assign_coords(lat_time=("y", time_dt))
        elif time_dt.ndim == 2:
            ds = ds.assign_coords(pixel_time=(("y", "x"), time_dt))
        else:
            ds["time_raw"] = (("y",), time_raw.ravel())  # last-resort fallback

        # derive pixel area from lat/lon bounds
        R = 6371e3  # Earth radius [m]

        latb = np.deg2rad(ds["lat_bounds"])
        lonb = np.deg2rad(ds["lon_bounds"])

        # reorder corners into closed polygon
        lat_poly = np.concatenate([latb, latb[..., :1]], axis=-1)
        lon_poly = np.concatenate([lonb, lonb[..., :1]], axis=-1)

        # Shoelace-like spherical area approximation
        area = 0.5 * np.abs(
            np.sum(
                (lon_poly[..., 1:] - lon_poly[..., :-1]) *
                (np.sin(lat_poly[..., 1:]) + np.sin(lat_poly[..., :-1])),
                axis=-1
            )
        ) * R**2    # m^2

        area_cm2 = area * 1e4

        # ds.assign_coords()
        ds = ds.assign_coords(pixel_area=(("y", "x"), area_cm2))
        # ds['pixel_area'] = (("y", "x"), area_cm2)

        return ds

    
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
    
    granules = earthaccess.search_data(
        temporal=(time.floor('D'), time.ceil('D')),
        short_name="TEMPO_NO2_L2",
        bounding_box=(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25),
        )

    files = [g['meta']['native-id'] for g in granules]

    p = sorted(
        [(f, datetime.datetime.strptime(re.search(r'_(\d{8}T\d{6})Z_', f).group(1), "%Y%m%dT%H%M%S"))
            for f in files],
        key=lambda x: x[1]
    )

    before = [f for f,t in p if t <= time]
    after  = [f for f,t in p if t >  time]
    filenames = (before[-1:] + after[:3])

    # List to store matching files
    all_files = []

    # Loop through each offset in the time range
    for f in filenames:
        # Find files matching the pattern for this hour
        # files = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2/TEMPO_NO2_L3_V03*{hour_str}*')
        files = sorted(glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/V04/L2/NO2/{f}')) \
            or sorted(glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/V03/L2/NO2/{f}'))
        # Append the found files to the list
        all_files.extend(files)

    # Open the dataset
    try:
        # ds = xr.open_mfdataset(all_files, group='product', engine=TEMPOBackendEntrypoint)
        ds = xr.open_mfdataset(all_files, group='product', engine=TEMPOBackendEntrypoint_L2, combine="nested", concat_dim="time")
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

    # only swaths that contain the launch point
    # compute scene bounds for each time
    lat_min = ds.latitude.min(dim=("y","x"))
    lat_max = ds.latitude.max(dim=("y","x"))
    lon_min = ds.longitude.min(dim=("y","x"))
    lon_max = ds.longitude.max(dim=("y","x"))

    # boolean mask for times containing launch point
    mask = (
        (lat >= lat_min) &
        (lat <= lat_max) &
        (lon >= lon_min) &
        (lon <= lon_max)
    )

    # filter dataset
    ds = ds.isel(time=mask.compute())
    
    bounding_box = (lon - 1, lon + 1, lat - 1, lat + 1)
    lon_min, lon_max, lat_min, lat_max = bounding_box

    # limit the dataset to the bounding box, and subtract the 10th percentile to get enhancements
    ds = mask_and_trim_to_roi(ds, bounding_box)
    # ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), drop=True)

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
    try:
        ds = load_tempo_data(time, lat, lon)
    except:
        print(f"raw TEMPO mask empty: (no pixels in bounds)")
    
    # Calculate the emissions
    rows = []
    # first try the data for the hour of the launch, then try the data for the hour after the launch
    for hour_after_launch in range(1,3):
        try:
            data = ds.isel(time=hour_after_launch)['vertical_column_troposphere']
        except:
            print(f"No data for {hour_after_launch} hours after launch", flush=True)
            continue

        # data = data.chunk({'latitude': -1, 'longitude': -1})
        # enhancements = data - data.quantile(0.1)
        enhancements = data - data.quantile(background_pct)

        vals_big = enhancements.values
        finite_big = np.isfinite(vals_big)
        global_med = np.nanmedian(vals_big[finite_big])
        global_mad = np.nanmedian(np.abs(vals_big[finite_big] - global_med))
        global_sigma = 1.4826*global_mad if global_mad > 0 else np.nanstd(vals_big[finite_big])

        plume = enhancements.copy(deep=True)

        # more strict bounding box for plume search
        bounding_box = (lon - 0.6, lon + 0.6, lat - 0.6, lat + 0.6)
        # bounding_box = (lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5)
        # bounding_box = (lon - 1, lon + 1, lat - 1, lat + 1)
        lon_min, lon_max, lat_min, lat_max = bounding_box
        # plume = plume.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), drop=True)
        
        try:
            plume = mask_and_trim_to_roi(plume, bounding_box)
        except:
            print(f"ROI mask empty: bbox={bounding_box} (no pixels in bounds)")
            continue

        # check that there are non nan values in plume
        if np.isnan(plume).all():
            print("No data", flush=True)
            continue

        # check that there are enough non nan values in plume
        if np.isnan(plume).sum() / plume.size > 0.5:
            print("Too many nans", flush=True)
            continue

        # Detect the plume
        threshold = enhancements.quantile(plume_pct).values
        print(f"threshold: {threshold:.2e}, global med: {global_med:.2e}, global sigma: {global_sigma:.2e}", flush=True)
        if threshold > 3.5e15:
            print("threshold too high, skipping", flush=True)
            continue

        try:
            # bounding_box = (lon - 0.25, lon + 0.25, lat - 0.25, lat + 0.25)
            bounding_box = (lon - 0.3, lon + 0.3, lat - 0.3, lat + 0.3)
            detected_plume = detect_single_plume(plume, bounding_box, (lat, lon), threshold, min_plume_size=10, snr_threshold=4.09, max_busy_frac=0.2,)
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
                fig, ax = plt.subplots(figsize=(4,3), dpi=200, subplot_kw={'projection': ccrs.PlateCarree(),})
                plot_TEMPO_L2(plume, ax=ax, vmin=-8e15, vmax=8e15, cmap='RdBu_r')
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
        # plume_pixel_area_1d = pixel_area_cm2(isolated_plume.latitude)
        # Broadcast to 2D
        # plume_pixel_area_2d = xr.broadcast(plume_pixel_area_1d, isolated_plume)[0]

        plume_pixel_area_2d = isolated_plume['pixel_area'].values
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
            fig, axs = plt.subplots(1,2, figsize=(8,3), dpi=100, sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree(),})
            # im0 = plume.plot(ax=axs[0], vmin=-8e15, vmax=8e15, cmap='RdBu_r')
            im0 = plot_TEMPO_L2(plume, ax=axs[0], vmin=-8e15, vmax=8e15, cmap='RdBu_r', show_colorbar=True)
            im1 = plot_TEMPO_L2(detected_plume, ax=axs[1], vmin=0, vmax=1, cmap='viridis')
            # detected_plume.plot(ax=axs[1],)
            # im0.colorbar.set_label(r'$\Delta$NO$_2$ [molec/cm$^2$]')
            axs[0].set_title(f"$\Delta NO2$ \n {time}")
            axs[1].set_title(f"Detected plume \n NOx mass = {ime:.2f} kg")
            axs[1].set_axis_off();
            fig.tight_layout();
            # save the figure
            # plt.show()
            plt.savefig(f"{outpath}/figs/detections/{time.strftime('%Y%m%dT%H')}_hour{hour_after_launch}.png")
            plt.close()

        # get centroid of plume
        yy,xx = np.where(isolated_plume)
        centroid_y, centroid_x = isolated_plume.y[yy].values.mean(), isolated_plume.x[xx].values.mean()
        centroid_lat = isolated_plume.latitude.sel(y=centroid_y, x=centroid_x, method='nearest').values
        centroid_lon = isolated_plume.longitude.sel(y=centroid_y, x=centroid_x, method='nearest').values
        obs_time = isolated_plume.lat_time.sel(y=centroid_y, method='nearest').values
        time_delta = obs_time - time

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
        row['obs_time'] = obs_time
        row['time_delta'] = time_delta
        row['time_delta'] = pd.to_timedelta(row['time_delta'])
        row['centroid_lat'] = centroid_lat
        row['centroid_lon'] = centroid_lon
        # row['obs_time'] = data.time.values
        # row['time_delta'] = data.time.values - time
        # row['time_delta'] = pd.to_timedelta(row['time_delta'])

        rows.append(row)
    
    if len(rows)>0:
        return pd.DataFrame(rows)
    else:
        print("No plume detected", flush=True)
        return

