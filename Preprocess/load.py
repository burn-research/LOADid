# -*- coding: utf-8 -*-
"""
Created on Apr 26

@author: hanssens
"""

#%% Helpers

import os
import pandas as pd
from tqdm import tqdm
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COL_RENAME_MAP = {
    # Gas
    'Datum': 'DATE',
    'Datum_Startuur': 'START_TIME',
    'Datum_Einduur': 'END_TIME',
    'Volume_Afname_KWh': 'CONS',
    'Type_Gasmeter': 'METER_TYPE',
    'Contract_Categorie': 'SECTOR', # Residential only (as recored throughout the year)
    # Elec
    'Volume_Injectie_KWh': 'INJ',
    'Warmtepomp_Indicator': 'HP', # Designated as the main heating system
    'Elektrisch_Voertuig_Indicator': 'EV', # Charging point reported by the customer (or EV detected)
    'PV_Installatie_Indicator': 'PV', # If a feed-in volume has been recorded over the past 12 months
}

#%% Load Profiles

def _load_csvs(folder: str) -> pd.DataFrame:
    """
    Load all CSV files from the specified folder and combines them into a single DataFrame.
    """
    all_files = os.listdir(folder)
    df_list = []
    
    for file in tqdm(all_files, desc='Loading CSV files'):
        if file.endswith('.csv'):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path)
            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def _localise_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Localise the specified date columns in the DataFrame to the CET/CEST time zone.
    """
    for col in date_columns:
        if col in df.columns:
            # Remove mislabeled UTC timezone (assumed to be CET/CEST as per documentation) and convert to naive datetime
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=False).dt.tz_localize(None)
    return df

def _aggregate_daily(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Group the DataFrame by the specified columns and aggregate the consumption and injection values.
    """
    agg_kwargs = {}
    for col in ['CONS', 'INJ']:
        if col in df.columns:
            agg_kwargs[f'{col}_sum'] = (col, 'sum')
            agg_kwargs[f'{col}_mean'] = (col, 'mean')
            agg_kwargs[f'{col}_median'] = (col, 'median')
            agg_kwargs[f'{col}_q25'] = (col, lambda x: x.quantile(0.25))
            agg_kwargs[f'{col}_q75'] = (col, lambda x: x.quantile(0.75))

    other_cols = [c for c in df.columns if c not in group_cols and c not in ('CONS', 'INJ')]
    for col in other_cols:
        agg_kwargs[col] = (col, 'first')

    df_grouped = df.groupby(group_cols, as_index=False).agg(**agg_kwargs)
    return df_grouped

def extract_daily_fluvius_data(elec_folder: str) -> pd.DataFrame:
    """
    Extracts electricity or gas data from the specified folder.
    """
    # Load the CSV files and combine them into a single DataFrame
    t0 = time.time()
    print(f"Loading CSV files from {elec_folder}...")
    combined_df = _load_csvs(elec_folder)
    print(f"Loaded {len(combined_df)} rows in {time.time() - t0:.2f} seconds.")

    # Convert date columns to datetime (CET/CEST time zone)
    print("Localising date columns...")
    t1 = time.time()
    date_columns = ['Datum', 'Datum_Startuur', 'Datum_Einduur']
    combined_df = _localise_dates(combined_df, date_columns)
    print(f"Date columns localised in {time.time() - t1:.2f} seconds.")

    # Rename columns
    col_rename_map = {k: v for k, v in COL_RENAME_MAP.items() if k in combined_df.columns}
    combined_df.rename(columns=col_rename_map, inplace=True)

    # Group by EAN_ID and date columns (daily resolution)
    print("Aggregating data to daily resolution...")
    t2 = time.time()
    group_cols = ['EAN_ID', 'DATE']
    df_grouped = _aggregate_daily(combined_df, group_cols)
    print(f"Data aggregated to daily resolution in {time.time() - t2:.2f} seconds.")

    # Rename values in SECTOR
    df_grouped['SECTOR'] = df_grouped['SECTOR'].replace({'Residentieel': 'RES'})

    return df_grouped

#%% Main

if __name__ == "__main__":
    elec_folder = os.path.join(BASE_DIR, '../Data/Elec/Raw')
    df_elec = extract_daily_fluvius_data(elec_folder)
    df_elec.to_feather(os.path.join(BASE_DIR, '../Data/Elec/elec_daily.feather'))

    gas_folder = os.path.join(BASE_DIR, '../Data/Gas/Raw')
    df_gas = extract_daily_fluvius_data(gas_folder)
    df_gas.to_feather(os.path.join(BASE_DIR, '../Data/Gas/gas_daily.feather'))
