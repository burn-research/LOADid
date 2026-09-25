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

df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../Data/Meteo/meteo_daily.feather'))
df_day = pd.read_feather(os.path.join(BASE_DIR, '../Data/day.feather'))

df_res_gas = pd.read_pickle(os.path.join(BASE_DIR, 'Gas/regression_results.pkl'))
df_res_elec = pd.read_pickle(os.path.join(BASE_DIR, 'Elec/regression_results.pkl'))

df_res_gas_mlp = pd.read_pickle(os.path.join(BASE_DIR, 'Gas/mlp_results.pkl'))
df_res_elec_mlp = pd.read_pickle(os.path.join(BASE_DIR, 'Elec/mlp_results.pkl'))
#%% Cross-residual analysis

df_res_gas_m = df_res_gas[df_res_gas['MODEL'] == 'M3']
mean_res_gas = np.mean(np.array(df_res_gas_m['res'].values.tolist()), axis=0)

mean_res_gas_mlp = np.mean(np.array(df_res_gas_mlp['res'].values.tolist()), axis=0)

df_res_elec_m = df_res_elec[df_res_elec['MODEL'] == 'M3']
df_res_elec_m_system = df_res_elec_m[(df_res_elec_m['HP'] == 1) & (df_res_elec_m['PV'] == 0) & (df_res_elec_m['EV'] == 0)]
mean_res_elec = np.mean(np.array(df_res_elec_m_system['res'].values.tolist()), axis=0)

df_res_elec_mlp_system = df_res_elec_mlp[(df_res_elec_mlp['HP'] == 1) & (df_res_elec_mlp['PV'] == 0)  & (df_res_elec_mlp['EV'] == 0)]
mean_res_elec_mlp = np.mean(np.array(df_res_elec_mlp_system['res'].values.tolist()), axis=0)

t = df_day['DATE'].values

# Regression

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3)

nwd = df_day['NWD'].values

ax[0].plot(t, mean_res_gas, color='orange', lw=2, label='Gas')
ax[0].plot(t,mean_res_elec, color='steelblue', lw=2, label='Elec')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Mean residuals')

ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[0].tick_params(axis='x', rotation=30)

ax[0].legend()

ax[1].scatter(mean_res_gas[nwd == 1], mean_res_elec[nwd == 1], color='k', alpha=0.5, label='Working days')
ax[1].scatter(mean_res_gas[nwd == 0], mean_res_elec[nwd == 0], color='tomato', alpha=0.5, label='Non-working days')
ax[1].set_xlabel('Mean residuals Gas')
ax[1].set_ylabel('Mean residuals Elec')

ax[1].legend()

corr_coef = np.corrcoef(mean_res_gas, mean_res_elec)[0, 1]
ax[1].text(0.95, 0.05, f'Correlation: {corr_coef:.2f}', transform=ax[1].transAxes, ha='right', va='bottom', fontsize=12)

plt.savefig(os.path.join(BASE_DIR, f'../Figures/Comparison/elec_gas_reg.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../Figures/Comparison/elec_gas_reg.pdf'), bbox_inches='tight')

plt.show()

# MLP

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3)

nwd = df_day['NWD'].values

ax[0].plot(t, mean_res_gas_mlp, color='orange', lw=2, label='Gas')
ax[0].plot(t, mean_res_elec_mlp, color='steelblue', lw=2, label='Elec')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Mean residuals')

ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[0].tick_params(axis='x', rotation=30)

ax[0].legend()

ax[1].scatter(mean_res_gas_mlp[nwd == 1], mean_res_elec_mlp[nwd == 1], color='k', alpha=0.5, label='Working days')
ax[1].scatter(mean_res_gas_mlp[nwd == 0], mean_res_elec_mlp[nwd == 0], color='tomato', alpha=0.5, label='Non-working days')
ax[1].set_xlabel('Mean residuals Gas')
ax[1].set_ylabel('Mean residuals Elec')

ax[1].legend()

corr_coef = np.corrcoef(mean_res_gas_mlp, mean_res_elec_mlp)[0, 1]
ax[1].text(0.95, 0.05, f'Correlation: {corr_coef:.2f}', transform=ax[1].transAxes, ha='right', va='bottom', fontsize=12)

plt.savefig(os.path.join(BASE_DIR, f'../Figures/Comparison/elec_gas_mlp.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../Figures/Comparison/elec_gas_mlp.pdf'), bbox_inches='tight')

plt.show()

# %%
