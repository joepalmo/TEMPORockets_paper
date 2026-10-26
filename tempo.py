import xarray as xr
from xarray.backends import BackendEntrypoint
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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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

            # viewing geometry
            solar_zenith_angle = file_id["geolocation"]["solar_zenith_angle"][:].filled(np.nan)
            viewing_zenith_angle = file_id["geolocation"]["viewing_zenith_angle"][:].filled(np.nan)
            
            albedo = file_id.groups["support_data"]["albedo"][:].filled(np.nan)
            ps = file_id["support_data"]["surface_pressure"][:].filled(np.nan)

            temperature_profile = file_id.groups["support_data"]["temperature_profile"][:].filled(np.nan)
            temperature_profile = np.flip(temperature_profile, axis=2)

            eta_a = file_id["support_data"]["surface_pressure"].Eta_A[:]
            eta_b = file_id["support_data"]["surface_pressure"].Eta_B[:]
            p_edge = eta_a[:, None, None] + eta_b[:, None, None] * ps[None, :, :]
            p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
            p_mid = np.transpose(p_mid, (1, 2, 0))
            p_mid = np.flip(p_mid, axis=2)

            scattering_weights = file_id.groups["support_data"]["scattering_weights"][:].filled(np.nan)
            scattering_weights = np.flip(scattering_weights, axis=2)
            gas_profile = file_id.groups["support_data"]["gas_profile"][:].filled(np.nan)
            gas_profile = np.flip(gas_profile, axis=2)

            support_data = file_id.groups["support_data"]
            if "scattering_weights_clear_sky" in support_data.variables:
                scattering_weights_clear_sky = support_data["scattering_weights_clear_sky"][:].filled(np.nan)
                scattering_weights_clear_sky = np.flip(scattering_weights_clear_sky, axis=2)
                amf_trop_clear_sky = support_data["amf_troposphere_clear_sky"][:].filled(np.nan)
            else:
                scattering_weights_clear_sky = np.full_like(scattering_weights, np.nan)
                amf_trop_clear_sky = np.full(lat_2d.shape, np.nan)

            amf_total = file_id.groups["support_data"]["amf_total"][:].filled(np.nan)
            amf_trop = file_id.groups["support_data"]["amf_troposphere"][:].filled(np.nan)
            amf_strat = file_id.groups["support_data"]["amf_stratosphere"][:].filled(np.nan)
            tropopause_pressure = file_id.groups["support_data"]["tropopause_pressure"][:].filled(np.nan)
            slant_column = file_id.groups["support_data"]["fitted_slant_column"][:].filled(np.nan)
            cloud_fraction = file_id["support_data"]["eff_cloud_fraction"][:].filled(np.nan)
            cloud_pressure = file_id["support_data"]["amf_cloud_pressure"][:].filled(np.nan)

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
                "solar_zenith_angle": (("y", "x"), solar_zenith_angle),
                "viewing_zenith_angle": (("y", "x"), viewing_zenith_angle),
                "albedo": (("y", "x"), albedo),
                "scattering_weights": (("y", "x", "lev"), scattering_weights),
                "scattering_weights_clear_sky": (("y", "x", "lev"), scattering_weights_clear_sky),
                "gas_profile": (("y", "x", "lev"), gas_profile),
                "surface_pressure": (("y", "x"), ps),
                "amf_total": (("y", "x"), amf_total),
                "amf_troposphere": (("y", "x"), amf_trop),
                "amf_troposphere_clear_sky": (("y", "x"), amf_trop_clear_sky),
                "amf_stratosphere": (("y", "x"), amf_strat),
                "tropopause_pressure": (("y", "x"), tropopause_pressure),
                "slant_column": (("y", "x"), slant_column),
                "pressure_levels": (("y", "x", "lev"), p_mid),
                "temperature_profile": (("y", "x", "lev"), temperature_profile),
                "cloud_pressure": (("y", "x"), cloud_pressure),
            },
            coords={
                "y": np.arange(ny),
                "x": np.arange(nx),
                "latitude": (("y", "x"), lat_2d),
                "longitude": (("y", "x"), lon_2d),
                "lev": np.arange(1, 73, 1),
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

def plot_TEMPO_L2(ds, quantity="vertical_column_troposphere", show_colorbar=True, ax=plt.gca(), **kwargs):
    from scipy.ndimage import distance_transform_edt

    def fill_nan_nearest(a):
        a = np.asarray(a)
        bad = ~np.isfinite(a)
        if not bad.any():
            return a
        _, inds = distance_transform_edt(bad, return_indices=True)
        return a[tuple(inds)]

    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)
    if type(ds) == xr.Dataset:
        C = np.asarray(ds[quantity].values)
    elif type(ds) == xr.DataArray:
        C = np.asarray(ds.values)
    else:
        raise ValueError("ds must be xarray Dataset or DataArray")

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(C)

    lon_plot = fill_nan_nearest(lon)
    lat_plot = fill_nan_nearest(lat)

    # still mask C where invalid
    Cmask = np.ma.masked_where(~valid, C)

    m = ax.pcolormesh(
        lon_plot, lat_plot, Cmask,
        shading="auto", 
        transform=ccrs.PlateCarree(), **kwargs
    )
    if show_colorbar:
        plt.colorbar(m, ax=ax, label=r'$\Delta$NO$_2$ [molec/cm$^2$]')
    ax.coastlines()