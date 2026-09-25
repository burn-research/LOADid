#%% Helpers

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from scipy import stats

import joblib

plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_elec = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Elec/elec_daily.feather'))
df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/meteo_daily.feather'))
df_meteo_2024 = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/openmeteo_daily_2024.feather'))
df_meteo_2050 = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/openmeteo_daily_2050.feather'))

## Assign system type

def determine_system_type(df):
    """
    Assigns a system type based on the presence of PV, HP, and EV indicators.    
    """
    SYSTEM_DICT = {
        1: 'Only PV',
        2: 'Nothing',
        3: 'Only HP with PV',
        4: 'Only HP without PV',
        5: 'Only EV with PV',
        6: 'Only EV without PV',
        7: 'Only HP and EV with PV',
        8: 'Only HP and EV without PV'
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

#%% Modelling - 2024

MODEL_DEFS = {
    'M0': dict(param_names=['y0', 'alpha', 'Th', 'rho'], # Linear Self-PV
               smooth=False, lse=False, pv_type='linear'),
    'M1': dict(param_names=['y0', 'alpha', 'Th', 'rho', 'tau'], # Saturated Self-PV
               smooth=False, lse=False, pv_type='exp'),
    'M2': dict(param_names=['y0', 'alpha', 'Th', 'w', 'rho', 'tau'], # Thermal inertia
               smooth=True, lse=False, pv_type='exp'),
    'M3': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma', 'rho', 'tau'], # Solar Heat Gain
               smooth=True, lse=False, pv_type='exp'),
    'M4': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma', 'omega', 'rho', 'tau'], # Wind chill
               smooth=True, lse=False, pv_type='exp'),
    'M5': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma', 'omega', 'k', 'rho', 'tau'], # Smooth transition regime
               smooth=True, lse=True, pv_type='exp')
}

def smooth(x, w, n):
    """
    Applies a moving average smoothing to the input array x with a window size n and a weighting factor w.
    """
    weights = (1 / w) ** np.arange(n + 1)
    weights /= weights.sum() # Normalise weights

    x_smooth = x.copy()
    conv = np.convolve(x, weights)[:len(x)]
    x_smooth[n:] = conv[n:]

    return x_smooth

def heat(x, p, smooth_flag, lse_flag):
    """
    Compute the heating component of the model based on temperature, solar radiation, and wind speed.
    """
    T, G, W = x

    gamma = p.get('gamma', 0)
    Tb = p['Th'] - gamma * G

    T_eff = smooth(T, w=p['w'], n=2) if smooth_flag else T
    dT = Tb - T_eff

    if lse_flag:
        k = p['k']
        hdd = (1 / k) * np.logaddexp(0, k * dT)
    else:
        hdd = np.maximum(0, dT)

    omega = p.get('omega', 0)
    plc = p['alpha'] + omega * W

    return plc * hdd

def model(x, params, heating_flag, pv_flag, param_names, pv_type, smooth_flag, lse_flag):
    """
    Compute the predicted consumption from the grid based on the model parameters and input features.
    """
    T, G, W = x
    p = dict(zip(param_names, params))

    # Base component
    y_cst = p['y0'] * np.ones_like(T)

    # Heating component
    if heating_flag:
        y_hp = heat(x, p, smooth_flag, lse_flag)

    else:
        y_hp = 0

    # PV component
    if pv_flag:
        if pv_type == 'linear':
            y_pv = p['rho'] * G
        elif pv_type == 'exp':
            y_pv = p['rho'] * (1 - np.exp(-G / p['tau']))

    else:
        y_pv = 0

    y = y_cst + y_hp - y_pv

    return y

def objective_function(
        params, x, y, 
        heating_flag, pv_flag,
        prior, prior_scale,
        model_fn
):
    """
    Compute the objective function associated to Maximum A Posteriori (MAP) estimation for the model parameters.
    """
    pred = model_fn(x, params, heating_flag, pv_flag)
    res = y - pred
    # Sum of Squared Errors (SSE)
    sse = np.sum(res**2)
    # Profiled Negative Log-likelihood (NLL) for Gaussian errors
    n = len(res)
    sigma2 = np.var(res, ddof=0) # ddof to check
    nll = n / 2 * np.log(2 * np.pi * sigma2) + sse / (2 * sigma2) # - np.sum(stats.norm.logpdf(res, loc=0, scale=np.std(res, ddof=0)))
    # MAP (Gaussian prior)
    nlprior = 1 / 2 * np.sum(((params - prior) / prior_scale)**2)
    nlposterior = nll + nlprior
    # Loss function
    J = nlposterior
    return J

def get_param_spec(T, G, y, pv_type):
    """
    Set initial guesses, scales, and bounds for the model parameters based on the input features and consumption data.
    """
    dy = np.percentile(y, 95) - np.percentile(y, 5) # Robust range of y

    specs = {
        'y0': (np.percentile(y, 50), dy / 2, (1e-6, None)), # Init, Scale, Bounds
        'alpha': (0, dy / (18 - T.min()), (0, None)),
        'Th':    (18, 5, (T.min(), T.max())),
        'gamma': (0, 3 / G.mean(), (0, 1 / G.mean())),
        'omega': (0, dy / (18 - T.min()) * 1 / W.mean(), (0, None)),
        'k':     (1, 10, (1e-6, 10)),
        'w':     (2, 10, (1e-6, 100)),
        'tau':   (np.median(G), np.median(G), (1e-6, G.max())),
    }

    if pv_type == 'linear':
        specs['rho'] = (0, dy / (2 * G.max()), (0, None))
    else:
        specs['rho'] = (0, dy / 2, (0, 1))

    return specs

def get_x0_xs_bounds(T, G, y, param_names, pv_type):
    """
    Get initial guesses, scales, and bounds for the model parameters based on the input features and consumption data.
    """
    table = get_param_spec(T, G, y, pv_type)
    x0 = [table[p][0] for p in param_names]
    xs = [table[p][1] for p in param_names]
    bounds = [table[p][2] for p in param_names]

    return x0, xs, bounds

# Features
T = df_meteo['TEMP_AVG'].values
G = df_meteo['GHI'].values
W = df_meteo['WIND_SPEED_10M'].values

T_2024 = df_meteo_2024['temperature_2m'].values
G_2024 = df_meteo_2024['shortwave_radiation'].values
W_2024 = df_meteo_2024['wind_speed_10m'].values

t = df_elec['DATE'].values

x = (T_2024, G_2024, W_2024)

results = [] # Store results for all models and customers

from functools import partial

for model_id, cfg in tqdm(MODEL_DEFS.items(), desc='Models', unit='model'):

    # Define the model function with fixed parameters using partial
    model_fn = partial(
        model,
        param_names=cfg['param_names'],
        pv_type=cfg['pv_type'],
        smooth_flag=cfg['smooth'],
        lse_flag=cfg['lse']
    )

    for customer, df_customer in tqdm(df_elec.groupby('EAN_ID'), desc='Customers', unit='customer'):

        # Impose 1 for classification purposes
        heat_flag = 1 # df_customer['HP'].iloc[0] == 1
        pv_flag = 1 # df_customer['PV'].iloc[0] == 1

        y = df_customer['CONS_sum'].values

        if y.max() == 0:
            continue  # Skip customers with zero consumption

        y_norm = y / y.max()  # Normalise consumption

        x0, xs, bounds = get_x0_xs_bounds(T_2024, G_2024, y_norm, cfg['param_names'], cfg['pv_type'])

        prior, prior_scale = x0, xs

        result = minimize(
            objective_function,
            x0=x0,  # Initial guess
            args=(x, y_norm, heat_flag, pv_flag, prior, prior_scale, model_fn),
            bounds=bounds,
            method='L-BFGS-B'
        )

        params_opt = result.x
        loss_opt = result.fun

        y_pred = model_fn(x, params_opt, heat_flag, pv_flag)
        res = y_norm - y_pred

        sse = np.sum(res**2)
        sst = np.sum((y_norm - np.mean(y_norm))**2)
        r2 = 1 - sse / sst
        rmse = np.sqrt(np.mean(res**2))

        # Compute AIC and BIC
        k = len(params_opt)
        n = len(y_norm)
        aic = n * np.log(sse / n) + 2 * k
        bic = n * np.log(sse / n) + k * np.log(n)

        opt_dict = dict(zip(cfg['param_names'], np.column_stack((prior, params_opt)))) # Prior and posterior

        acf_res = acf(res, nlags=15)

        results.append({
            'MODEL': model_id,
            'SYSTEM': df_customer['SYSTEM'].iloc[0],
            'SYSTEM_LBL': df_customer['SYSTEM_LBL'].iloc[0],
            'EAN_ID': customer,
            'HP': df_customer['HP'].iloc[0],
            'PV': df_customer['PV'].iloc[0],
            'EV': df_customer['EV'].iloc[0],
            'true': y_norm,
            'pred': y_pred,
            'ymax': y.max(),
            'loss_opt': loss_opt,
            'res': res,
            'acf_res': acf_res,
            'R2': r2,
            'RMSE': rmse,
            'AIC': aic,
            'BIC': bic,
            'y0': opt_dict.get('y0', np.nan),
            'alpha': opt_dict.get('alpha', np.nan),
            'Th': opt_dict.get('Th', np.nan),
            'gamma': opt_dict.get('gamma', np.nan),
            'omega': opt_dict.get('omega', np.nan),
            'k': opt_dict.get('k', np.nan),
            'w': opt_dict.get('w', np.nan),
            'rho': opt_dict.get('rho', np.nan),
            'tau': opt_dict.get('tau', np.nan),
            'sigma': np.std(res),
            'acf1': acf_res[1], # Day before
            'acf7': acf_res[7], # Last week
        })

df_results = pd.DataFrame(results)
#%% Projection - 2050

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

# Keep only posterior estimates
for param in valid_params:
    df_results_m[param] = df_results_m[param].apply(lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x)

df_hp = df_results_m[(df_results_m['HP'] == 1) & (df_results_m['PV'] == 0) & (df_results_m['EV'] == 0)].copy()
df_none = df_results_m[(df_results_m['HP'] == 0) & (df_results_m['PV'] == 0) & (df_results_m['EV'] == 0)].copy()

# Define model and heat functions with fixed parameters using partial
model_fn = partial(
    model,
    param_names=MODEL_DEFS[m]['param_names'],
    pv_type=MODEL_DEFS[m]['pv_type'],
    smooth_flag=MODEL_DEFS[m]['smooth'],
    lse_flag=MODEL_DEFS[m]['lse']
)

heat_fn = partial(
    heat,
    smooth_flag=MODEL_DEFS[m]['smooth'],
    lse_flag=MODEL_DEFS[m]['lse']
)

# 2024

y_hp = []; sh_hp = []
for customer in df_hp['EAN_ID'].unique():
    params = df_hp[df_hp['EAN_ID'] == customer][valid_params]
    p = dict(zip(valid_params, params.values[0]))

    y_c = model_fn((T_2024, G_2024, W_2024), list(p.values()), heating_flag=True, pv_flag=True)
    y_heat = heat_fn((T_2024, G_2024, W_2024), p)

    y_hp.append(y_c)
    sh_hp.append(y_heat)

y_hp_mean = np.mean(y_hp, axis=0)
sh_hp_sum = np.sum(sh_hp, axis=1)

y_none = []; sh_none = []
for customer in df_none['EAN_ID'].unique():
    params = df_none[df_none['EAN_ID'] == customer][valid_params]
    p = dict(zip(valid_params, params.values[0]))

    y_c = model_fn((T_2024, G_2024, W_2024), list(p.values()), heating_flag=True, pv_flag=True)
    y_heat = heat_fn((T_2024, G_2024, W_2024), p)

    y_none.append(y_c)
    sh_none.append(y_heat)

y_none_mean = np.mean(y_none, axis=0)
sh_none_sum = np.sum(sh_none, axis=1)

# 2050

T_2050 = df_meteo_2050['temperature_2m_mean'].values
G_2050 = df_meteo_2050['shortwave_radiation_sum'].values * 1e6/3600 / 24 # Convert from sum of MJ/m² to mean W/m² over the day
W_2050 = df_meteo_2050['wind_speed_10m_mean'].values

def projected_params(p, renovation_rate=0.01, renovation_depth=2, sobriety_adjustment=0, heat_pump_adoption=1):
    """
    Adjusts the model parameters to account for renovation, sobriety, and heat pump adoption effects.
    """
    # Renovation
    if rng.random() < (1 - (1 - renovation_rate) ** 25):
        p['alpha'] /= renovation_depth

    # Sobriety
    p['Th'] -= sobriety_adjustment

    # Heat pump adoption
    p['alpha'] *= heat_pump_adoption

    return p

# Run the projection for each scenario

SCENARIOS = {
    'Climate change & HP adoption': dict(renovation_rate=0, renovation_depth=1, sobriety_adjustment=0),
    '+ Renovation (BAU)': dict(renovation_rate=0.01, renovation_depth=2, sobriety_adjustment=0),
    '+ Renovation (WAM)': dict(renovation_rate=0.03, renovation_depth=2, sobriety_adjustment=0),
    '+ Sobriety': dict(renovation_rate=0.03, renovation_depth=2, sobriety_adjustment=1),
}

results = [] # Store results for each scenario

for scenario, params in SCENARIOS.items():
    y_hp_2050 = []; sh_hp_2050 = []
    rng = np.random.default_rng(seed=42)
    for customer in df_hp['EAN_ID'].unique():
        params_customer = df_hp[df_hp['EAN_ID'] == customer][valid_params]
        p = dict(zip(valid_params, params_customer.values[0]))

        p_new = projected_params(p, **params)

        y_c = model_fn((T_2050, G_2050, W_2050), list(p_new.values()), heating_flag=True, pv_flag=True)
        y_heat = heat_fn((T_2050, G_2050, W_2050), p_new)

        y_hp_2050.append(y_c)
        sh_hp_2050.append(y_heat)

    y_hp_2050_mean = np.mean(y_hp_2050, axis=0)
    sh_hp_2050_sum = np.sum(sh_hp_2050, axis=1)


    y_none_2050 = []; sh_none_2050 = []
    for customer in df_none['EAN_ID'].unique():
        params_customer = df_none[df_none['EAN_ID'] == customer][valid_params]
        p = dict(zip(valid_params, params_customer.values[0]))

        params.update({'heat_pump_adoption': 3})

        p_new = projected_params(p, **params)

        y_c = model_fn((T_2050, G_2050, W_2050), list(p_new.values()), heating_flag=True, pv_flag=True)
        y_heat = heat_fn((T_2050, G_2050, W_2050), p_new)

        y_none_2050.append(y_c)
        sh_none_2050.append(y_heat)

    y_none_2050_mean = np.mean(y_none_2050, axis=0)
    sh_none_2050_sum = np.sum(sh_none_2050, axis=1)

    # Compute the percentage change in space heating demand between 2024 and 2050 for both scenarios
    ratio_hp = 100 * np.divide((sh_hp_2050_sum - sh_hp_sum), sh_hp_sum, out=np.zeros_like(sh_hp_sum), where=sh_hp_sum!=0)
    ratio_none = 100 * np.divide((sh_none_2050_sum - sh_none_sum), sh_none_sum, out=np.zeros_like(sh_none_sum), where=sh_none_sum!=0)

    ratio_hp_mean = np.mean(ratio_hp); ratio_hp_std = np.std(ratio_hp)
    ratio_none_mean = np.mean(ratio_none); ratio_none_std = np.std(ratio_none)

    results.append({
        'scenario': scenario,
        'ratio_hp_mean': ratio_hp_mean,
        'ratio_hp_std': ratio_hp_std,
        'ratio_none_mean': ratio_none_mean,
        'ratio_none_std': ratio_none_std
    })

scenario_results = pd.DataFrame(results)

# Plot the last scenario results

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(wspace=0.3)

axs[0].scatter(T_2024, y_hp_mean, color='k', alpha=0.1, s=20, label='2024')
axs[1].scatter(T_2024, y_none_mean, color='k', alpha=0.1, s=20, label='2024')


axs[0].scatter(T_2050, y_hp_2050_mean, color='tomato', alpha=0.2, s=20, label=r'$\overline{2050}$')
axs[1].scatter(T_2050, y_none_2050_mean, color='tomato', alpha=0.2, s=20, label=r'$\overline{2050}$')

axs[0].set_title('Only HP')
axs[1].set_title('None')

axs[0].text(
    0.05, 0.90,
    rf"${ratio_hp_mean:+.1f}\%$ space heating demand",
    transform=axs[0].transAxes,
    verticalalignment='top'
)

axs[1].text(
    0.05, 0.90,
    rf"${ratio_none_mean:+.1f}\%$ space heating demand",
    transform=axs[1].transAxes,
    verticalalignment='top'
)

for ax in axs:
    ax.set_xlabel(r'Outside Temperature $\left[^{\circ}C\right]$')
    ax.set_ylabel('Mean Normalised Electricity Consumption')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Projection/sh_change_scatter.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Projection/sh_change_scatter.pdf'), bbox_inches='tight')

plt.show()

#%% Change in SH by scenario

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

x = np.arange(len(scenario_results))
width = 0.25

ax.bar(x - width/2, scenario_results['ratio_hp_mean'], width, yerr=scenario_results['ratio_hp_std'], color='tomato', error_kw={'ecolor': 'tomato'}, alpha=0.7, label='Only HP')
ax.set_yticks(np.arange(-60, 1, 20))
ax.tick_params(axis='y', colors='tomato')

for i, v in enumerate(scenario_results['ratio_hp_mean']):
    ax.text(i - width, v - 10, rf"{v:.0f}\%", color='tomato', ha='center', va='top', fontsize=10, fontweight='bold')

ax2 = ax.twinx()
ax2.bar(x + width/2, scenario_results['ratio_none_mean'], width, yerr=scenario_results['ratio_none_std'], color='steelblue', error_kw={'ecolor': 'steelblue'}, alpha=0.7, label='None')
ax2.set_yticks(np.arange(0, 251, 50))
ax2.tick_params(axis='y', colors='steelblue')

for i, v in enumerate(scenario_results['ratio_none_mean']):
    ax2.text(i + width*1.1, v + 20, rf"+{v:.0f}\%", color='steelblue', ha='center', va='top', fontsize=10, fontweight='bold')

ax.axhline(0, color='k', alpha=0.5, lw=1, ls='--')

ymin = min(scenario_results['ratio_hp_mean'].min(), scenario_results['ratio_none_mean'].min()) * 2
ymax = max(scenario_results['ratio_hp_mean'].max(), scenario_results['ratio_none_mean'].max()) * 1.8
ax.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.75, 1))
ax2.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.75, 0.9))

ax.set_xticks(x)
scenario_names = scenario_results['scenario'].str.replace('&', r'\\ \&')

ax.set_xticklabels(scenario_names, rotation=30, ha='center', fontsize=11)
ax.set_ylabel(r'Change in Space Heating Consumption [\%]')

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Projection/sh_change_scenario.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Projection/sh_change_scenario.pdf'), bbox_inches='tight')

plt.show()

#%% 2024 vs 2050

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(T_2024, color='k')
ax.plot(T_2050, color='tomato')

plt.show()

# %%
