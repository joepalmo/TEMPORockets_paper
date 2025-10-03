import xarray as xr
import pandas as pd
import netCDF4 as nc
import time
import numpy as np
from pathlib import Path
import re
import datetime

import warnings
warnings.filterwarnings("ignore")

### Some key datasets
# load GEOS pressure levels
pressure_levels_df = pd.read_csv("/home/jpalmo/fs09/Projects/TEMPO_apriori/code/geos_pressure_levels.csv")
# fuel burn from Cassandre paper, NOx EFs from NAS paper
emis_profile_df = pd.read_csv("/home/jpalmo/fs09/Projects/TEMPO_rocket_launches/F9 mass-layer emissions.csv")
emis_profile_df['pressure'] = pressure_levels_df['P(hPa)'].values[::-1]
# troposphere only 100 hPa cutoff
trop_emis_profile = emis_profile_df[emis_profile_df['pressure']>=100]


### FUNCTIONS

def load_geoscf(mode="assim", collection='chm_inst_1hr_g1440x721_v72', max_retries=3):
    """
    Load GEOS-CF data via OpeNDAP.
    - OpenDAP URL for GEOS-CF data
    https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/
    - See documentation for details:
    https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/

    - Collections include:
    chm_inst_1hr_g1440x721_p23
    chm_tavg_1hr_g1440x721_v1
    chm_inst_1hr_g1440x721_v72 ********* CHEMISTRY
    htf_inst_15mn_g1440x721_x1
    met_inst_1hr_g1440x721_p23
    met_inst_1hr_g1440x721_v72 ********* METEOROLOGY
    met_tavg_1hr_g1440x721_x1
    xgc_tavg_1hr_g1440x721_x1
    """
    # mode is either assim, fcast

    # Root OPeNDAP directory
    url = 'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/{}/{}'.format(
        mode, collection)

    for attempt in range(max_retries):
        try:
            # Open the dataset
            ds = xr.open_dataset(url, engine='netcdf4')

            # Assign these as the new coordinates for the 'lev' dimension
            ds = ds.assign_coords(lev=pressure_levels_df['P(hPa)'])
            return ds
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

def L_geoscf(geoscf_ds, time, lat, lon):

    nearest = geoscf_ds.sel(lat=lat, lon=lon, time=time, method='nearest')

    L = (nearest['no'] + nearest['no2']) / nearest['no2']

    # normalized fuel burn profile

    return ((trop_emis_profile['NOX-kg'] / trop_emis_profile['NOX-kg'].sum()) * L.sel(lev=slice(100, 1000))).sum()

def time_from_lat(ds, lat_pt):
    # representative latitude for each scanline (median over lon)
    lat_line = ds['lat'].median('lon')              # (lat,)
    i = int(np.abs(lat_line - lat_pt).argmin())     # row index
    return ds['time'].isel(time=i).values          # Python datetime

def heatmap_centroid(plume_ds):
    # Extract the 2D data array
    data = plume_ds["__xarray_dataarray_variable__"].values
    lat = plume_ds["latitude"].values
    lon = plume_ds["longitude"].values

    # Create 2D coordinate grids
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Avoid NaNs (replace with 0 weight)
    weights = np.nan_to_num(data, nan=0.0)

    # Weighted average for centroid
    total_weight = np.sum(weights)
    if total_weight == 0:
        return None  # no data

    centroid_lat = np.sum(lat2d * weights) / total_weight
    centroid_lon = np.sum(lon2d * weights) / total_weight

    return centroid_lat, centroid_lon

def _extract_ts(path: Path) -> datetime.datetime | None:
    # match  YYYYMMDDTHH[MM[SS]] anywhere in the filename
    m = re.search(r'(\d{8}T\d{2}(?:\d{2}(?:\d{2})?)?)', path.name)
    if not m:
        return None
    s = m.group(1)
    fmt = {11: "%Y%m%dT%H", 13: "%Y%m%dT%H%M", 15: "%Y%m%dT%H%M%S"}[len(s)]
    return datetime.datetime.strptime(s, fmt)

def next_file_after(dirpath: str, prefix: str, target_dt64: np.datetime64) -> str | None:
    target_dt = target_dt64.astype("datetime64[s]").astype(datetime.datetime)
    files = sorted(Path(dirpath).glob(f"{prefix}*.nc"))
    cand = [(ts, p) for p in files if (ts := _extract_ts(p)) is not None and ts >= target_dt]
    if not cand:
        return None
    return min(cand, key=lambda x: x[0])[1].as_posix()