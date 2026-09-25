#%% Helpers

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_elec = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Elec/elec_daily.feather'))
df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/meteo_daily.feather'))

## Assign system type

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

#%% TS analysis

from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

acf_matrix = []
pacf_matrix = []
n_lags = 30

# Mean, median, IQR for each system type
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

plt.savefig(os.path.join(BASE_DIR, '../../Figures/elec_system.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../../Figures/elec_system.pdf'), bbox_inches='tight')

plt.show()

# ACF - PACF analysis for each consumer
for cons_id, group in df_elec.groupby('EAN_ID'):
    series = group['CONS_sum'].values
    if np.var(series) == 0:
        continue  # Skip constant series
    acf_vals = acf(series, nlags=n_lags); acf_matrix.append(acf_vals)
    pacf_vals = pacf(series, nlags=n_lags); pacf_matrix.append(pacf_vals)

acf_matrix = np.array(acf_matrix); pacf_matrix = np.array(pacf_matrix)
conf_int = stats.norm.ppf(1 - 0.05/2) * 1/np.sqrt(len(series))
n_consumers = acf_matrix.shape[0]

fig, axs = plt.subplots(2, 1,figsize=(9, 6))

# ACF boxplot
axs[0].boxplot(
    acf_matrix, positions=range(n_lags + 1), 
    flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
    patch_artist=True, boxprops=dict(facecolor='lightblue'),
    medianprops=dict(color='steelblue')
)

axs[0].fill_between(range(n_lags + 1), -conf_int, conf_int, color='lightgray', alpha=0.4)
axs[0].axhline(0, color='black', lw=1, ls='--')

# PACF boxplot
axs[1].boxplot(
    pacf_matrix, positions=range(n_lags + 1), 
    flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
    patch_artist=True, boxprops=dict(facecolor='lightblue'),
    medianprops=dict(color='steelblue')
)
axs[1].fill_between(range(n_lags + 1), -conf_int, conf_int, color='lightgray', alpha=0.4)
axs[1].axhline(0, color='black', lw=1, ls='--')

plt.savefig(os.path.join(BASE_DIR, '../../Figures/acf_elec.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../../Figures/acf_elec.pdf'), bbox_inches='tight')

plt.show()

#%% Features analysis

T_y = []; G_y = []; W_y = []

for group, df_system in df_elec[df_elec.SYSTEM == 4].groupby('EAN_ID'):
    T = df_meteo['TEMP_AVG'].values
    G = df_meteo['GHI'].values
    W = df_meteo['WIND_SPEED_10M'].values
    if np.var(df_system['CONS_sum'].values) == 0:
                continue  # Skip constant series
    y = df_system['CONS_sum'].values / df_system['CONS_sum'].max() # Normalise consumption
    T_y.extend(list(zip(T, y)))
    G_y.extend(list(zip(G, y)))
    W_y.extend(list(zip(W, y)))

T_y = np.array(T_y); G_y = np.array(G_y); W_y = np.array(W_y)

def plot_density(ax, x, y, cmap):
    """
    Plots a density contour plot using Gaussian KDE.
    """
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    kde = stats.gaussian_kde(np.vstack([x, y]))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    ax.contourf(X, Y, Z, levels=50, cmap=cmap)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

plot_density(axs[0], T_y[:, 0], T_y[:, 1], cmap='Oranges')
axs[0].set_xlabel(r'Temperature $\left[^{\circ}C\right]$')
axs[0].set_ylabel('Normalised Consumption')
plot_density(axs[1], G_y[:, 0], G_y[:, 1], cmap='Reds')
axs[1].set_xlabel(r'GHI $\left[\frac{W}{m^2}\right]$')
plot_density(axs[2], W_y[:, 0], W_y[:, 1], cmap='Purples')
axs[2].set_xlabel(r'Wind Speed $\left[\frac{m}{s}\right]$')

plt.savefig(os.path.join(BASE_DIR, '../../Figures/elec_features.png'), dpi=300, bbox_inches='tight')

plt.show()

# %%
