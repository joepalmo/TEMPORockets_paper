import numpy as np
import pandas as pd
import glob
import netCDF4 as nc

import warnings
warnings.filterwarnings("ignore")

#### Constants
# TEMPO FOV
lon_min = -168
lon_max = -13
lat_min = 14
lat_max = 73

#### planet4589.org data
df = pd.read_csv('https://planet4589.org/space/gcat/tsv/launch/launch.tsv', sep='\t',)
sites_df = pd.read_csv('https://planet4589.org/space/gcat/tsv/tables/sites.tsv', sep='\t')

#### FUNCTIONS
def clean_and_parse_date(date_str):
    try:
        # Remove invalid characters like "?"
        cleaned_date = date_str.replace('?', '').strip()

        # Check if the format includes seconds
        if len(cleaned_date.split()) == 4 and ':' in cleaned_date:
            return pd.to_datetime(cleaned_date, format='%Y %b %d %H%M:%S')
        # Check if the format includes hours and minutes
        elif len(cleaned_date.split()) == 4:
            return pd.to_datetime(cleaned_date, format='%Y %b %d %H%M')
        # Check if it's just year, month, day
        elif len(cleaned_date.split()) == 3:
            return pd.to_datetime(cleaned_date, format='%Y %b %d')
        # Check if it's just year, month
        elif len(cleaned_date.split()) == 2 and 'Q' not in cleaned_date:
            return pd.to_datetime(cleaned_date, format='%Y %b')
        # Check if it's just year, quarter
        elif len(cleaned_date.split()) == 2 and 'Q' in cleaned_date:
            # Extract the year for quarters
            return int(cleaned_date.split()[0])  # Return the year only
        # Check if it's just year
        elif len(cleaned_date.split()) == 1:
            return pd.to_datetime(cleaned_date, format='%Y')
        else:
            return np.nan  # Return NaN for invalid entries
    except Exception:
        return np.nan

# given the Launch_Site, find the corresponding lat and lon
def get_lat(site):
    try:
        return sites_df[sites_df['#Site'] == site]['Latitude'].values[0]
    except Exception:
        return np.nan
    
def get_lon(site):
    try:
        return sites_df[sites_df['#Site'] == site]['Longitude'].values[0]
    except Exception:
        return np.nan

# check if files were downloaded for specific launch
def check_files(launch):
    files = glob.glob(f'/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2/TEMPO_NO2_L3_V03*{launch}*')
    return len(files) > 0

#### DATA CLEANING

def get_launch_df():
    # Apply the cleaning and parsing function
    df['timestamp'] = df['Launch_Date'].apply(clean_and_parse_date)
    df.dropna(subset=['timestamp'], inplace=True)
    # set timestamp index
    df.set_index('timestamp', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)

    # only launches during TEMPO mission
    recent_df = df.loc['2023-08-01':]

    # get the lat and lon for each launch site
    recent_df['lat'] = recent_df['Launch_Site'].apply(get_lat).astype(float)
    recent_df['lon'] = recent_df['Launch_Site'].apply(get_lon).astype(float)

    tempo_df = recent_df[(recent_df['lon'] > lon_min) & (recent_df['lon'] < lon_max) & (recent_df['lat'] > lat_min) & (recent_df['lat'] < lat_max)]


    # make new dataframe only where files are available
    for t, row in tempo_df.iterrows():
        hour_str = t.strftime('%Y%m%dT%H')
        # print(hour_str, "-----", check_files(hour_str))
        tempo_df.loc[t, 'files'] = check_files(hour_str)

    tempo_df = tempo_df[tempo_df['files'] == True]

    tempo_df.sort_index(inplace=True)

    return tempo_df