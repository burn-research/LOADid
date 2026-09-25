#%% Helpers

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import openmeteo_requests


plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#%% 2025 Historical weather data

# Client
openmeteo = openmeteo_requests.Client()

# ~ 9 (ECMWF) - 25 (ERA5) km resolution (reanalysis datasets)
url = "https://archive-api.open-meteo.com/v1/archive"

# Antwerp, Belgium
lat = 51.2205
lon = 4.4003

# YYYY-MM-DD
start_date = '2024-01-01'
end_date = '2024-12-31'

hourly_vars = [
    # Temp
    'temperature_2m',
    # Solar (average over the past hour / instant at the indicated time)
    # shortwave = GHI, direct = Beam, diffuse = DHI, direct_normal_irradiance = DNI
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'direct_normal_irradiance',
    # Humidity
    'relative_humidity_2m',
    # Wind
    'wind_speed_10m', 'wind_direction_10m'
    # Precipitation
    # Other
]

params = {
    'latitude': lat,
    'longitude': lon,
    'hourly': hourly_vars, 
    'daily': None,
    'start_date': start_date,
    'end_date': end_date,
    'temperature_unit': 'celsius',
    'wind_speed_unit': 'ms',
    'precipitation_unit': 'mm',
    'timeformat': 'iso8601', # YYYY-MM-DDThh:mm:ss (T separates date and time)
    'timezone': 'auto', # GMT = UTC + 00:00 (<!> Standard: SDT vs Daylight Saving: DST)
    'models': 'best_match', # Best Match combines IFS HRES, ERA5, ERA5_Land (Select ERA5-Land for consistent data)
}

# Call the Weather API
# ~ 100 calls (max. 10.000 calls / day)
responses = openmeteo.weather_api(url, params=params)

response = responses[0]
# Location and times (for-loop for multiple locations or weather models)
print(f"Coordinates: {response.Latitude():.2f}°N {response.Longitude():.2f}°E")
print(f"Elevation: {response.Elevation()} m asl\n") # Above sea level

# Hourly
hourly = response.Hourly()

# Index
hourly_data = pd.DataFrame({
    "date": pd.date_range(
        start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit='s', utc=False).tz_localize(None),
        end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit='s', utc=False).tz_localize(None),
        freq = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left"
    )
})

# Variables
for vari in range(hourly.VariablesLength()):
    var = hourly.Variables(vari).ValuesAsNumpy()
    # The order of variables needs to be the same as requested
    name = hourly_vars[vari]
    hourly_data[name] = var

# Process data
df_meteo_hourly = hourly_data

df_meteo_daily = hourly_data.groupby(hourly_data.date.dt.date).agg(
    temperature_2m_min = ('temperature_2m', 'min'),
    temperature_2m_max = ('temperature_2m', 'max'),
    **{
        col: (col, 'mean') for col in hourly_vars
    }
).reset_index()

df_meteo_daily['date'] = pd.to_datetime(df_meteo_daily['date'], utc=False).dt.tz_localize(None)

# Save data
df_meteo_daily.to_feather(os.path.join(BASE_DIR, '../Data/Meteo/openmeteo_daily_2024.feather'))


# %% 2050 Forecast weather data

# Client
openmeteo = openmeteo_requests.Client()

# ~ 7 regional downscaled climate models (CMIP6 - HighResMip) from ERA5-Land - RCP8.5 (SSP5-8.5) scenario
url = "https://climate-api.open-meteo.com/v1/climate"

# Antwerp, Belgium
lat = 51.2205
lon = 4.4003

# YYYY-MM-DD
start_date = '2050-01-01'
end_date = '2050-12-31'

daily_vars = [
    # Temp
    'temperature_2m_mean',
    # Solar (average over the past hour / instant at the indicated time)
    # shortwave = GHI, direct = Beam, diffuse = DHI, direct_normal_irradiance = DNI
    'shortwave_radiation_sum',
    # Humidity
    'relative_humidity_2m_mean',
    # Wind
    'wind_speed_10m_mean'
    # Precipitation
    # Other
]

models = [
    'CMCC_CM2_VHR4',
    'FGOALS_f3_H',
    'HiRAM_SIT_HR',
    'MRI_AGCM3_2_S',
    'EC_Earth3P_HR',
    'MPI_ESM1_2_XR',
    'NICAM16_8S'
]

params = {
    'latitude': lat,
    'longitude': lon,
    'hourly': None, 
    'daily': daily_vars,
    'start_date': start_date,
    'end_date': end_date,
    'models': models,
    'temperature_unit': 'celsius',
    'wind_speed_unit': 'ms',
    'precipitation_unit': 'mm',
    'timeformat': 'iso8601', # YYYY-MM-DDThh:mm:ss (T separates date and time)
    'timezone': 'auto', # GMT = UTC + 00:00 (<!> Standard: SDT vs Daylight Saving: DST)
}

# Call the Weather API
# ~ 100 calls (max. 10.000 calls / day)
responses = openmeteo.weather_api(url, params=params)

df_meteo_daily = []

for response in responses:

    # Model, Location and times (for-loop for multiple locations or weather models)
    print(f"Model: {response.Model()}")
    print(f"Coordinates: {response.Latitude():.2f}°N {response.Longitude():.2f}°E")
    print(f"Elevation: {response.Elevation()} m asl\n") # Above sea level

    # Hourly
    daily = response.Daily()

    # Index
    daily_data = pd.DataFrame({
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit='s', utc=False).tz_localize(None),
            end =  pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit='s', utc=False).tz_localize(None),
            freq = pd.Timedelta(seconds=daily.Interval()),
            inclusive = "left"
        )
    })

    # Variables
    for vari in range(daily.VariablesLength()):
        var = daily.Variables(vari).ValuesAsNumpy()
        # The order of variables needs to be the same as requested
        name = daily_vars[vari]
        daily_data[name] = var

    daily_data['model'] = response.Model()

    df_meteo_daily.append(daily_data)

df_meteo_daily = pd.concat(df_meteo_daily, ignore_index=True)

# take the mean of the models for comparison with 2025 historical data
df_meteo_daily_mean = df_meteo_daily.groupby('date').mean().reset_index().drop(columns='model')

# Save data
df_meteo_daily_mean.to_feather(os.path.join(BASE_DIR, '../Data/Meteo/openmeteo_daily_2050.feather'))

# %%
