#%% Helpers

import os
import pandas as pd

import holidays
from datetime import datetime, timedelta
from astral import LocationInfo
from astral.sun import sun

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_non_working_days(df, country='BE', date_column='DATE', weekend={5, 6}):
    """
    Identify non-working days in the DataFrame based on the specified date column.    
    """
    df = df.copy()

    dates = pd.to_datetime(df[date_column])

    df['WEEKEND'] = dates.dt.weekday.isin(weekend)
    holiday_calendar = holidays.country_holidays(country, years=dates.dt.year.unique())
    df['HOLIDAY'] = dates.isin(holiday_calendar)
    df['NWD'] = df['WEEKEND'] | df['HOLIDAY']
    df['WD'] = ~df['NWD']

    return df

def compute_daylengths(df, date_column='DATE'):
    """
    Compute daylengths for each day in the DataFrame based on the specified date column.    
    """
    df = df.copy()

    city = LocationInfo("Brussels", "Belgium", "Europe/Brussels", 50.85, 4.35)
    daylengths = []

    for date in df[date_column]:
        s = sun(city.observer, date=date)
        daylength = (s['sunset'] - s['sunrise']).total_seconds() / 3600.0  # in hours
        daylengths.append(daylength)

    df['DAYLENGTH'] = daylengths
    df['NIGHTLENGTH'] = 24 - df['DAYLENGTH']

    return df

# %% Main

if __name__ == "__main__":

    start_date = '2024-01-01'
    end_date = '2024-12-31'

    # Create a DataFrame with all days in the specified range
    df_day = pd.date_range(start=start_date, end=end_date, freq='D').to_frame(index=False, name='DATE')

    # Compute non-working days
    df_day = get_non_working_days(df_day, country='BE', date_column='DATE')

    # Compute daylengths
    df_day = compute_daylengths(df_day, date_column='DATE')

    # Save results
    df_day.to_feather(os.path.join(BASE_DIR, '../Data/day.feather'))
# %%
