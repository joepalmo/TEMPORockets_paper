import requests
import earthaccess
import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd

# TEMPO FOV
lon_min = -168
lon_max = -13
lat_min = 14
lat_max = 73

df = pd.read_csv('https://planet4589.org/space/gcat/tsv/launch/launch.tsv', sep='\t',)
sites_df = pd.read_csv('https://planet4589.org/space/gcat/tsv/tables/sites.tsv', sep='\t')

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

# Apply the cleaning and parsing function
df['timestamp'] = df['Launch_Date'].apply(clean_and_parse_date)
df.dropna(subset=['timestamp'], inplace=True)
# set timestamp index
df.set_index('timestamp', inplace=True)
df.index = pd.DatetimeIndex(df.index)
df.sort_index(inplace=True)

# only launches during TEMPO mission
recent_df = df.loc['2025-08-01':]

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

# get the lat and lon for each launch site
recent_df['lat'] = recent_df['Launch_Site'].apply(get_lat).astype(float)
recent_df['lon'] = recent_df['Launch_Site'].apply(get_lon).astype(float)

tempo_df = recent_df[(recent_df['lon'] > lon_min) & (recent_df['lon'] < lon_max) & (recent_df['lat'] > lat_min) & (recent_df['lat'] < lat_max)]

#############################################
# Download TEMPO data

# Access the values of the command line arguments
# to download L3 data
short_name = "TEMPO_NO2_L3"
output_dir = "/home/jpalmo/fs09/Datasets/TEMPO/L3/NO2"
# to download L2 data
# short_name = "TEMPO_NO2_L2"
# output_dir = "/home/jpalmo/fs09/Datasets/TEMPO/L2/NO2"

# Log in to Earthdata
earthaccess.login()

for time,launch in tempo_df.iterrows():
    start = (time - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    end = (time + pd.Timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')

    # create a small 2°x2° box around the launch site
    box = (launch['lon'] - 1,  launch['lat'] - 1, launch['lon'] + 1, launch['lat'] + 1)

    results = earthaccess.search_data(
    short_name=short_name,
    temporal=(start, end),
    # if L2 data, use bounding box
    # bounding_box=box
    )

    if len(results)>=1:
        # Download the data
        files = earthaccess.download(results, output_dir)



