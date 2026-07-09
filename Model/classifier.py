# -*- coding: utf-8 -*-
"""
Created on May 26

@author: hanssens
"""

#%% Helpers

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_elec = pd.read_feather(os.path.join(BASE_DIR, '../Data/Elec/elec_daily.feather'))
df_gas = pd.read_feather(os.path.join(BASE_DIR, '../Data/Gas/gas_daily.feather'))
df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../Data/Meteo/meteo_daily.feather'))

#%% Electricity Profiles

def determine_system_type(df):
    """
    Assigns a system type based on the presence of PV, HP, and EV indicators.    
    """
    SYSTEM_DICT = {
        1: 'Only PV', # PV (1)
        2: 'Nothing', # No PV no HP (1)
        3: 'Only HP with PV', # PV and HP (1)
        4: 'Only HP without PV', # HP (1)
        5: 'Only EV with PV', # PV (2)
        6: 'Only EV without PV', # No PV no HP (2)
        7: 'Only HP and EV with PV', # PV and HP (2)
        8: 'Only HP and EV without PV' # HP (2)
    }

    # TODO: Reduce the size of the study by removing the EV part

    mask_pv = df['PV'] == 1
    mask_hp = df['HP'] == 1
    mask_ev = df['EV'] == 1

    df['SYSTEM'] = 0  # Init
    df.loc[mask_pv & ~mask_hp & ~mask_ev, 'SYSTEM'] = 1
    df.loc[~mask_pv & ~mask_hp & ~mask_ev, 'SYSTEM'] = 2
    df.loc[mask_pv & mask_hp & ~mask_ev, 'SYSTEM'] = 3
    df.loc[~mask_pv & mask_hp & ~mask_ev, 'SYSTEM'] = 4
    df.loc[mask_pv & ~mask_hp & mask_ev, 'SYSTEM'] = 5
    df.loc[~mask_pv & ~mask_hp & mask_ev, 'SYSTEM'] = 6
    df.loc[mask_pv & mask_hp & mask_ev, 'SYSTEM'] = 7
    df.loc[~mask_pv & mask_hp & mask_ev, 'SYSTEM'] = 8

    df['SYSTEM_LBL'] = df['SYSTEM'].map(SYSTEM_DICT)

    return df
    
df_elec = determine_system_type(df_elec)

#%% Plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

fig, axs = plt.subplots(2, 4, figsize=(24, 8))
for group, df_group in df_elec.groupby('SYSTEM'):
    ax = axs.flatten()[group-1]

    var = 'CONS_sum'
    df_ensemble = df_group.groupby('DATE').agg(
        mean = (var, 'mean'),
        median = (var, 'median'),
        std = (var, 'std'),
        q25 = (var, lambda x: x.quantile(0.25)),
        q75 = (var, lambda x: x.quantile(0.75))
    ).reset_index()

    ax.plot(df_ensemble['DATE'], df_ensemble['mean'], color='steelblue', lw=2, label='Mean')
    ax.plot(df_ensemble['DATE'], df_ensemble['median'], color='steelblue', lw=1, ls='--', label='Median')
    ax.fill_between(df_ensemble['DATE'], df_ensemble['q25'], df_ensemble['q75'], color='steelblue', alpha=0.2, label='IQR')

    ax.set_ylim(bottom=0)

    group_lbl = df_group['SYSTEM_LBL'].iloc[0]
    ax.legend(title=group_lbl, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30); fig.subplots_adjust(hspace=0.35)

    if group > 4:
        ax.set_xlabel('Date')
    if group in [1, 5]:
        ax.set_ylabel(r'Electricity Consumption $\left[\frac{kWh}{day}\right]$')

plt.savefig(os.path.join(BASE_DIR, '../Figures/elec_system.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../Figures/elec_system.pdf'), bbox_inches='tight')

plt.show()

#%% Classification on features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def HDD(T, base=15, setpoint=18):
    """
    Calculates the Heating Degree Days (HDD) for a given temperature.
    HDD = max(0, setpoint - T) if T < base else 0
    """
    HDD = np.where(T < base, np.maximum(0, setpoint - T), 0)
    return HDD

T = df_meteo['TEMP_AVG'].values
G = df_meteo['GHI'].values

features = []
for group, customer in tqdm(df_elec[['SYSTEM', 'EAN_ID']].drop_duplicates().values, desc="Processing customers"):
    df_customer = df_elec[df_elec['EAN_ID'] == customer].copy()
    
    y = df_customer['CONS_sum'].values
    X = np.column_stack([HDD(T), G])
    X_scaled = StandardScaler().fit_transform(X)

    reg = LinearRegression().fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)
    res = y - y_pred

    features.append({
        'SYSTEM': group,
        'EAN_ID': customer,
        'INTERCEPT': reg.intercept_,
        'HDD_COEF': reg.coef_[0],
        'IRRADIANCE_COEF': reg.coef_[1],
        'RES_STD': np.std(res)
        # PEAK_DAILY_CONS ?
        # MAX_DAILY_CONS ?
        # WEEKEND_WEEKDAY_GAP (AVG of 52 values) ?
    })
    
F = pd.DataFrame(features)
yf = F['SYSTEM'].values
Xf = F.drop(columns=['SYSTEM', 'EAN_ID']).values
Xf_scaled = StandardScaler().fit_transform(Xf)

Xf_train, Xf_test, yf_train, yf_test, idx_train, idx_test = train_test_split(Xf_scaled, yf, F.index, test_size=0.2, stratify=yf, random_state=42)
F['TRAIN_TEST'] = ['Train' if i in idx_train else 'Test' for i in F.index]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xf_train, yf_train)

# Predict for all customers
F['PREDICTED_SYSTEM'] = clf.predict(Xf_scaled)

# Attach predictions to the original DataFrame
df_elec_classified = df_elec.merge(
    F[['EAN_ID', 'TRAIN_TEST', 'PREDICTED_SYSTEM']],
    on='EAN_ID', how='left'
)

# Metrics
print(classification_report(yf_test, clf.predict(Xf_test)))
# clf.feature_importances_

# Confusion matrix
groups = df_elec.sort_values('SYSTEM')['SYSTEM_LBL'].unique()
cm = confusion_matrix(yf_test, clf.predict(Xf_test), labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=groups)
disp.plot(cmap='Blues', xticks_rotation=90, colorbar=False)
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()

# %%

# TODO: Analyse the coefficients of the regression in F (features) for each system:
# Is it possible to propose an automated classification approach without any training ? 
# M1 : L = f(HDD, G) + u, M2 : L = f(HDD) + u, M3 : L = f(G) + u -> ANOVA
# Take into account the fact that we know the injection for each user: We know the PV component

# TODO: Compare different techniques on different training ratio:
# 0% = automatic, 10%, 20%, ..., 80%

# TODO: Based on the classification model, select the relevant model (M1, M2, M3) and estimate the heating consumption
# The PV self-consumption should be distinguished from the heating component, to avoid surestimation

# TODO: Compare the thermal-sensitivity coefficient with gas consumption

# TODO: Apply bayesian regression to estimate the thermal-sensitivity coefficient and the heating threshold

# TODO: Use a metric that 