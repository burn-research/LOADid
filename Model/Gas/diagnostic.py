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

df_gas = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Gas/gas_daily.feather'))
df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/meteo_daily.feather'))

#%% TS analysis

from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

acf_matrix = []
pacf_matrix = []
n_lags = 30

# Mean, median, IQR for each system type
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, (group, df_group) in enumerate(df_gas.groupby('METER_TYPE')):
    ax = axs.flatten()[i]

    var = 'CONS_sum'
    df_ensemble = df_group.groupby('DATE').agg(
        mean = (var, 'mean'),
        median = (var, 'median'),
        std = (var, 'std'),
        q25 = (var, lambda x: x.quantile(0.25)),
        q75 = (var, lambda x: x.quantile(0.75))
    ).reset_index()

    ax.plot(df_ensemble['DATE'], df_ensemble['mean'], color='orange', lw=2, label='Mean')
    ax.plot(df_ensemble['DATE'], df_ensemble['median'], color='orange', lw=1, ls='--', label='Median')
    ax.fill_between(df_ensemble['DATE'], df_ensemble['q25'], df_ensemble['q75'], color='orange', alpha=0.2, label='IQR')

    ax.set_ylim(bottom=0)

    group_lbl = df_group['METER_TYPE'].iloc[0]
    ax.legend(title=group_lbl, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30); fig.subplots_adjust(hspace=0.35)

    ax.set_xlabel('Date')
    if i == 0:
        ax.set_ylabel(r'Gas Consumption $\left[\frac{m^3}{day}\right]$')

plt.savefig(os.path.join(BASE_DIR, '../../Figures/gas_system.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../../Figures/gas_system.pdf'), bbox_inches='tight')

plt.show()

# ACF - PACF analysis for each consumer
for cons_id, group in df_gas.groupby('EAN_ID'):
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
    patch_artist=True, boxprops=dict(facecolor='bisque'),
    medianprops=dict(color='orange')
)

axs[0].fill_between(range(n_lags + 1), -conf_int, conf_int, color='lightgray', alpha=0.4)
axs[0].axhline(0, color='black', lw=1, ls='--')

# PACF boxplot
axs[1].boxplot(
    pacf_matrix, positions=range(n_lags + 1), 
    flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
    patch_artist=True, boxprops=dict(facecolor='bisque'),
    medianprops=dict(color='orange')
)
axs[1].fill_between(range(n_lags + 1), -conf_int, conf_int, color='lightgray', alpha=0.4)
axs[1].axhline(0, color='black', lw=1, ls='--')

plt.savefig(os.path.join(BASE_DIR, '../../Figures/acf_gas.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../../Figures/acf_gas.pdf'), bbox_inches='tight')

plt.show()

#%% Features analysis

T_y = []; G_y = []; W_y = []

for group, df_system in df_gas[df_gas.METER_TYPE == 'G4'].groupby('EAN_ID'):
    T = df_meteo['TEMP_AVG'].values
    G = df_meteo['GHI'].values
    W = df_meteo['WIND_SPEED_10M'].values
    if np.var(df_system['CONS_sum'].values) == 0:
            continue  # Skip constant series
    y = df_system['CONS_sum'].values / df_system['CONS_sum'].max()  # Normalise consumption
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

plt.show()

# %%
