# -*- coding: utf-8 -*-
"""
Created on Apr 26

@author: hanssens
"""

#%% Helpers

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import io
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#%% Population Density

def get_eurostat_population_density(dataset_code='demo_r_d3dens'):
    """
    Download population density by NUTS3 region for a given year from Eurostat,
    using the SDMX-CSV format (flat, no JSON-stat parsing needed).
    User Guide: https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started/api
    """
    # Construct the Eurostat API URL
    EUROSTAT_URL = f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_code}'
    params = {
        'format': 'SDMX-CSV',
        'lang': 'en',
    }
    # GET request to Eurostat API
    r = requests.get(EUROSTAT_URL, params=params)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    
    cols_to_keep = {
        'geo' : 'NUTS_ID',
        'TIME_PERIOD' : 'YEAR',
        'OBS_VALUE' : 'POP_DENSITY',
        'OBS_FLAG' : 'OBS_FLAG',
        'unit' : 'UNIT',
    }
    df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)

    return df

def get_gisco_nuts(version=2016, geometry_type='RG', scale='10M', crs='4326'):
    """
    Load NUTS data for a specific year from the GISCO distribution API.
    API: https://gisco-services.ec.europa.eu/distribution/v2/
    Geometry Type: 'RG' = regions (polygons), 'BN' = boundaries (lines), 'LB' = labels (points)
    Scale: '01M' (most detailed) ... '60M' (least detailed)
    CRS: '4326' = WGS84
    """
    # GISCO data distribution API
    BASE_URL = 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson'
    file_name = f'NUTS_{geometry_type}_{scale}_{version}_{crs}.geojson'
    url = f'{BASE_URL}/{file_name}'

    r = requests.get(url)
    r.raise_for_status()

    gdf = gpd.read_file(r.content)
    gdf['VERSION'] = int(version) # Track version

    cols_to_keep = ['NUTS_ID', 'LEVL_CODE', 'CNTR_CODE', 'NUTS_NAME', 'NAME_ENGL', 'VERSION', 'geometry']
    gdf = gdf[cols_to_keep]

    return gdf

def get_flanders_population_density(year=2024):
    """
    Get population density for Flanders by merging Eurostat population density data with GISCO NUTS geometries.
    """
    crs_lambert = 'EPSG:31370'  # Belgian Lambert 72
    gdf = get_gisco_nuts().to_crs(crs_lambert)
    df = get_eurostat_population_density()

    gdf_flanders = gdf[gdf['NUTS_ID'].str.startswith('BE2') & (gdf['LEVL_CODE'] == 3)].copy()
    nuts_ids = gdf_flanders['NUTS_ID'].unique()
    years = range(df['YEAR'].min(), df['YEAR'].max() + 1)

    # Build full (NUTS_ID, YEAR) grid so missing combinations are included
    full_grid = pd.MultiIndex.from_product([nuts_ids, years], names=['NUTS_ID', 'YEAR'])
    df_full = df.set_index(['NUTS_ID', 'YEAR']).reindex(full_grid).reset_index()
    df_full = df_full.sort_values(['NUTS_ID', 'YEAR'])

    # Track, for each row, which year its value actually comes from
    source_year = df_full['YEAR'].where(df_full['POP_DENSITY'].notna())

    # Fill missing values per region with the nearest available year
    was_missing = df_full['POP_DENSITY'].isna()
    df_full['POP_DENSITY'] = df_full.groupby('NUTS_ID')['POP_DENSITY'].transform(lambda x: x.ffill().bfill())
    source_year = source_year.groupby(df_full['NUTS_ID']).transform(lambda x: x.ffill().bfill())

    # Flag filled rows with the year the value was taken from
    filled = was_missing & df_full['POP_DENSITY'].notna()
    df_full.loc[filled, 'OBS_FLAG'] = 'nearest_' + source_year[filled].astype(int).astype(str)

    # Merge with geometries and select year
    gdf_flanders = gdf_flanders.merge(df_full, on='NUTS_ID', how='left')

    # Select year
    gdf_flanders = gdf_flanders[gdf_flanders['YEAR'] == year].copy()

    # Add absolute population (kPER) by multiplying density with area (in km²)
    gdf_flanders['POP'] = gdf_flanders['POP_DENSITY'] * gdf_flanders['geometry'].area / 1e6 * 1e-3

    return gdf_flanders

gdf_flanders = get_flanders_population_density()

#%% Meteo

def get_rmi_data(layer='aws:aws_1day', start=2024, end=2024, gdf_bounds=None):
    """
    Get RMI data from the WFS service.
    """
    # Connect to the WFS service
    WFS_IRM = "https://opendata.meteo.be/service/ows"

    # Temporal filter
    start_date = f'{start}-01-01 00:00:00'
    end_date = f'{end}-12-31 23:59:59'
    time_filter = f"timestamp BETWEEN '{start_date}' AND '{end_date}'"

    # Spatial filter
    if gdf_bounds is not None:
        bounds_wgs84 = gdf_bounds.dissolve().to_crs('EPSG:4326').total_bounds
        bbox_str = "{},{},{},{}".format(*(bounds_wgs84[i] for i in (1, 0, 3, 2))) # miny, minx, maxy, maxx
        cql_filter = f"{time_filter} AND BBOX(the_geom, {bbox_str})"
    else:
        cql_filter = time_filter

    crs_lambert = 'EPSG:31370' # Belgian Lambert 72
    params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
        'typeNames': layer,
        'CQL_FILTER': cql_filter,
        'sortBy': 'timestamp desc',
        'outputFormat': 'application/json',
        'srsName': crs_lambert,
    }

    r = requests.get(WFS_IRM, params=params)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))

    cols_to_keep = {
        'code' : 'STATION',
        'timestamp' : 'DATE',
        'temp_avg' : 'TEMP_AVG',
        'temp_min' : 'TEMP_MIN',
        'temp_max' : 'TEMP_MAX',
        'wind_speed_10m' : 'WIND_SPEED_10M',
        'humidity_rel_shelter_avg' : 'HUMIDITY_REL',
        'sun_duration' : 'SUN_DURATION',
        'short_wave_from_sky_avg' : 'GHI',
        'sun_int_avg' : 'DNI',
        'geometry' : 'geometry',
    }
    gdf = gdf[list(cols_to_keep.keys())].rename(columns=cols_to_keep)

    # Remove timezone from DATE column
    gdf['DATE'] = pd.to_datetime(gdf['DATE'], errors='coerce', utc=False).dt.tz_localize(None)

    gdf = gdf.sort_values(['DATE', 'STATION']).reset_index(drop=True)

    return gdf

gdf_rmi = get_rmi_data(layer='aws:aws_1day', start=2024, end=2024, gdf_bounds=gdf_flanders)

#%% Average Meteo

def population_weighted_average_meteo(gdf_meteo, gdf_population):
    """
    Calculate the population-weighted average of meteorological data for each date.
    """
    # Station locations
    stations = gdf_meteo[['STATION', 'geometry']].drop_duplicates()

    # Assign each population polygon to its nearest station
    population_points = gdf_population.copy()
    population_points['geometry'] = population_points['geometry'].representative_point()

    assigned = gpd.sjoin_nearest(population_points[['POP', 'geometry']], stations, how='left')
    
    # Connect stations with assigned points
    station_geom = stations.set_index('STATION')['geometry']
    stations_connection = gpd.GeoDataFrame({
        'STATION': assigned['STATION'],
        'POP': assigned['POP'],
    }, 
    geometry=[LineString([station_geom[station], point]) for station, point in zip(assigned['STATION'], assigned['geometry'])], 
    crs=stations.crs
    )

    # Station weight = total population of all regions assigned to it
    station_weights = assigned.groupby('STATION')['POP'].sum().rename('WEIGHT')

    # Attach weights to the meteo data
    df_meteo = gdf_meteo.merge(station_weights, on='STATION', how='left')

    # Weighted average for each date
    value_cols = [c for c in df_meteo.columns if c not in ['STATION', 'DATE', 'geometry', 'WEIGHT']]

    def weighted_avg(group):
        weights = group['WEIGHT']
        return pd.Series({col: (group[col] * weights).sum() / weights.sum() for col in value_cols})
    
    df_meteo_avg = df_meteo.groupby('DATE').apply(weighted_avg).reset_index()

    return df_meteo_avg, stations_connection

def simple_average_meteo(gdf_meteo):
    """
    Calculate the simple average of meteorological data for each date.
    """
    value_cols = [c for c in gdf_meteo.columns if c not in ['STATION', 'DATE', 'geometry']]
    df_meteo_avg = gdf_meteo.groupby('DATE')[value_cols].mean().reset_index()
    return df_meteo_avg

df_meteo_wavg, stations_connection = population_weighted_average_meteo(gdf_rmi, gdf_flanders)
df_meteo_savg = simple_average_meteo(gdf_rmi)

#%% Plot
import matplotlib.pyplot as plt
plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

fig, axs = plt.subplots(2, 1, figsize=(12, 6))
# AX0
gdf_flanders.plot(ax=axs[0], column='POP', cmap='Reds')
cbar = fig.colorbar(axs[0].collections[0], ax=axs[0], orientation='vertical')
cbar.set_label('Population (kPER)')

stations = gdf_rmi[['STATION', 'geometry']].drop_duplicates().sort_values('STATION')
stations['WEIGHT'] = stations_connection.sort_values('STATION').groupby('STATION')['POP'].sum().values
stations.plot(ax=axs[0], color='k', markersize=stations['WEIGHT']/5)

axs[0].scatter([], [], color='k', label='AWS Stations')
stations_connection.plot(ax=axs[0], color='k', alpha=0.6, label='Associated Population')
axs[0].legend(loc='upper left', bbox_to_anchor=(-0.3, 0.9), frameon=False)
axs[0].set_axis_off()
# AX1
for station, gdf_station in gdf_rmi.groupby('STATION'):
    df_station = gdf_station.drop(columns='geometry')
    df_station.plot(ax=axs[1], x='DATE', y='TEMP_AVG', color='grey', alpha=0.4, label='_nolegend_')
df_meteo_savg.plot(ax=axs[1], x='DATE', y='TEMP_AVG', color='k', label='Simple average', linestyle='--')
df_meteo_wavg.plot(ax=axs[1], x='DATE', y='TEMP_AVG', color='orangered', label='Population-weighted', linestyle='--')

axs[1].set_xlabel('Date')
axs[1].set_ylabel('Average Temperature (°C)')

axs[1].plot([], [], color='grey', alpha=0.4, label='Individual Stations')
axs[1].legend()

plt.savefig(os.path.join(BASE_DIR, '../Figures/meteo_aws.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../Figures/meteo_aws.pdf'), bbox_inches='tight')

plt.show()

# %% Main

if __name__ == "__main__":
    # Get population density for Flanders
    gdf_flanders = get_flanders_population_density(year=2024)

    # Get RMI data for 2024 within Flanders bounds
    gdf_rmi = get_rmi_data(layer='aws:aws_1day', start=2024, end=2024, gdf_bounds=gdf_flanders)

    # Calculate population-weighted and simple average meteo data
    df_meteo_wavg, stations_connection = population_weighted_average_meteo(gdf_rmi, gdf_flanders)
    df_meteo_savg = simple_average_meteo(gdf_rmi)

    # Save results
    df_meteo_wavg.to_feather(os.path.join(BASE_DIR, '../Data/Meteo/meteo_daily.feather'))