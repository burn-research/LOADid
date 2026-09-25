#%% Helpers

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from scipy import stats

import joblib

plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_meteo_2025 = pd.read_feather(os.path.join(BASE_DIR, 'meteo_daily_2025.feather'))
df_meteo_2050 = pd.read_feather(os.path.join(BASE_DIR, 'meteo_daily_2050.feather'))

#%% Load Sibelga data

def parse_sibelga_date(date_str):
    """
    Convert Sibelga date format (e.g., '01JAN2025') to pandas datetime.
    """

    month_map = {
        'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
        'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'
    }

    day = date_str[:2] # %d
    month = month_map.get(date_str[2:5]) # %m
    year = date_str[5:] # %Y

    return pd.to_datetime(f"{day}{month}{year}", format="%d%m%Y")

def add_meter_id(df, date_col='DATES'):
    """
    Add a unique meter ID to each row in the DataFrame based on the date column.
    """
    df = df.copy()  

    # A new meter starts when the date goes backwards
    new_meter = df[date_col].lt(df[date_col].shift())

    df['METER_ID'] = new_meter.cumsum() + 1  # Start meter IDs from 1

    return df

# Load Sibelga electricity data
df_elec = pd.read_csv(os.path.join(BASE_DIR, '../Data/Brussels/BE_12_AMR_ELEC_2025.csv'), sep=';', decimal='.', encoding='latin-1', dtype='str')

# Filter for residential customers
df_elec = df_elec[df_elec['CUSTOMER_TYPE_FR_TX'] == 'RES']

# Parse dates and add meter IDs
df_elec['DATES'] = df_elec['DATES'].apply(parse_sibelga_date)
df_elec = add_meter_id(df_elec, date_col='DATES')

# Convert consumption to float
df_elec['VOLUME_KWH'] = df_elec['VOLUME_KWH'].astype(float)

# Plot load profiles for each meter
fig, ax = plt.subplots(figsize=(12, 5))

for i in df_elec['METER_ID'].unique():
    df_c = df_elec[df_elec['METER_ID'] == i]
    ax.plot(df_c['DATES'], df_c['VOLUME_KWH'], color='steelblue', lw=2, alpha=0.1)

df_mean = df_elec.groupby('DATES')['VOLUME_KWH'].mean().reset_index()
df_median = df_elec.groupby('DATES')['VOLUME_KWH'].median().reset_index()

ax.plot(df_mean['DATES'], df_mean['VOLUME_KWH'], color='steelblue', lw=2, label='Mean')
ax.plot(df_median['DATES'], df_median['VOLUME_KWH'], color='steelblue', lw=2, linestyle='--', label='Median')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=30); fig.subplots_adjust(hspace=0.35)

ax.legend()

plt.show()
#%% Modelling

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

        y_hp = plc* hdd

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
T = df_meteo_2025['temperature_2m'].values
G = df_meteo_2025['shortwave_radiation'].values
W = df_meteo_2025['wind_speed_10m'].values

t = df_elec['DATES'].values

x = (T, G, W)

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

    for customer, df_customer in tqdm(df_elec.groupby('METER_ID'), desc='Customers', unit='customer'):

        # Impose 1 for classification purposes
        heat_flag = 1
        pv_flag = 1

        y = df_customer['VOLUME_KWH'].values

        if y.max() == 0 or len(y) != len(T):
            continue # Skip customers with zero consumption

        y_norm = y / y.max() # Normalise consumption

        x0, xs, bounds = get_x0_xs_bounds(T, G, y_norm, cfg['param_names'], cfg['pv_type'])

        prior, prior_scale = x0, xs

        result = minimize(
            objective_function,
            x0=x0, # Initial guess
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
            'METER_ID': customer,
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

#%% Result analysis

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

fig = plt.figure(figsize=(12, 8))

gs = fig.add_gridspec(
    nrows=2, ncols=3,
    width_ratios=[1, 1, 1],
    wspace=0.5, hspace=0.3
)

gs_params = gs[:, 2].subgridspec(len(valid_params), 1, hspace=0.5)

ax11 = fig.add_subplot(gs[0, 0])
ax21 = fig.add_subplot(gs[1, 0])

gs_ax12 = gs[0, 1].subgridspec(2, 2, wspace=0.3, hspace=0.3)
ax12 = [fig.add_subplot(gs_ax12[i, j]) for i in range(2) for j in range(2)]

ax22 = fig.add_subplot(gs[1, 1])

axs_right = [fig.add_subplot(gs_params[i, 0]) for i in range(len(valid_params))]

# Residuals
res_mean = np.mean(np.array(df_results_m['res'].values.tolist()), axis=0)
res_quantiles = np.percentile(np.array(df_results_m['res'].values.tolist()), [25, 75], axis=0)

# Plot of mean residuals over time
ax11.plot(res_mean, color='k', lw=1)
ax11.fill_between(range(len(res_mean)), res_quantiles[0], res_quantiles[1], color='k', alpha=.08, edgecolor='none', label='IQR')
# Add legend without the boundary box
ax11.legend(frameon=False)
ax11.set_xlabel('Date')
ax11.set_ylabel('Mean residuals')

# Scatter plot of y, y_pred and residuals vs temperature
ax12[0].scatter(T, res_mean, color='k', alpha=0.3, s=6)
ax12[0].set_ylabel('Mean residuals')
ax12[0].set_xticks([])

ax12[1].scatter(G, res_mean, color='k', alpha=0.3, s=6)
ax12[1].set_xticks([]); ax12[1].set_yticks([])

ax12[2].scatter(T, df_results_m['true'].values.mean(), color='k', alpha=0.3, s=6)
ax12[2].scatter(T, df_results_m['pred'].values.mean(), color='tomato', alpha=0.3, s=6)
ax12[2].set_xlabel(r'Outside\\Temperature $\left[^{\circ}C\right]$')
ax12[2].set_ylabel('Mean Normalised\nconsumption')

ax12[3].scatter(G, df_results_m['true'].values.mean(), color='k', alpha=0.3, s=6)
ax12[3].scatter(G, df_results_m['pred'].values.mean(), color='tomato', alpha=0.3, s=6)
ax12[3].set_xlabel(r'GHI $\left[\frac{W}{m^2}\right]$')
ax12[3].set_yticks([])

# Boxplot of ACF values
acf_matrix = np.array(df_results_m['acf_res'].values.tolist())
conf_int = stats.norm.ppf(1 - 0.05/2) * 1/np.sqrt(len(res_mean))
ax21.boxplot(
    acf_matrix, positions=range(acf_matrix.shape[1]),
    showfliers=True, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
    patch_artist=True, boxprops=dict(facecolor='lightblue'),
    medianprops=dict(color='steelblue')
)
ax21.fill_between(range(acf_matrix.shape[1]), -conf_int, conf_int, color='lightgray', alpha=0.4)
ax21.axhline(0, color='black', lw=1, ls='--')
ax21.set_xlabel('Lag')
ticks = list(range(11)) + [14]
ax21.set_xticks(ticks)
ax21.set_xticklabels([str(x) for x in ticks])
ax21.set_ylabel('ACF')

# QQ plot
mean_sample_quantiles = np.sort(res_mean)
mean_theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(mean_sample_quantiles)))

ax22.scatter(mean_theoretical_quantiles, mean_sample_quantiles, color='k', alpha=0.2, s=20)
m = np.std(mean_sample_quantiles)
ax22.plot(mean_theoretical_quantiles, m * mean_theoretical_quantiles, color='k')
ax22.set_xlabel('Theoretical Quantiles')
ax22.set_ylabel('Mean Sample Quantiles')

# Parameters
for i, param in enumerate(valid_params):
    priors = [v[0] for v in df_results_m[param]]
    posteriors = [v[1] for v in df_results_m[param]]

    if np.var(priors) <= 1e-6:
        axs_right[i].scatter(priors[0], 1, s=20, facecolor='lightyellow', edgecolor='gold')
        axs_right[i].set_ylim(0.5, 2.5)
    else:
        bp_prior = axs_right[i].boxplot(
            priors, orientation='horizontal', positions = [1], widths=0.5,
            showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
            patch_artist=True, boxprops=dict(facecolor='lightyellow'),
            medianprops=dict(color='gold'),
        )

    bp_post = axs_right[i].boxplot(
        posteriors, orientation='horizontal', positions = [2], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green'),
    )

    axs_right[i].set_ylabel(param)
    axs_right[i].set_yticks([])

    if i == 0:
        legend_handles = [
            Patch(facecolor='lightyellow', edgecolor='gold', label='Prior'),
            Patch(facecolor='lightgreen', edgecolor='green', label='Posterior')
        ]
        axs_right[i].legend(handles=legend_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

plt.show()

#%% Features distribution

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plt.subplots_adjust(wspace=0.3)

# HP

alpha_values = [v[1] for v in df_results_m['alpha']]

bins = np.linspace(min(alpha_values), max(alpha_values), 20)

counts, edges = np.histogram(alpha_values, bins=bins) # Number of samples in each bin

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[0].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color='tomato',
    edgecolor='none',
    alpha=0.7
)

axs[0].set_xlabel(r'$\alpha$')

# PV

rho_values = [v[1] for v in df_results_m['rho']]

bins = np.linspace(min(rho_values), max(rho_values), 20)

counts, edges = np.histogram(rho_values, bins=bins) # Number of samples in each bin

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[1].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color = 'gold',
    edgecolor='none',
    alpha=0.7
)

axs[1].set_xlabel(r'$\rho$')

# EV

ymax_values = [v for v in df_results_m['ymax']]

bins = np.linspace(min(ymax_values), max(ymax_values), 20)

counts, edges = np.histogram(ymax_values, bins=bins) # Number of samples in each bin

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[2].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color='steelblue',
    edgecolor='none',
    alpha=0.7
)

axs[2].set_xlabel(r'$y_{max}$')
plt.show()

#%% System Identification

# Select model
m = 'M5'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

params = valid_params + ['ymax', 'acf1', 'acf7'] # Features for classification
X = df_results_m[params].map(lambda x: x if isinstance(x, (int, float)) else x[1]) # Use posterior values when relevant

scaler = joblib.load(os.path.join(BASE_DIR, f'scaler.pkl'))
clf_rf = joblib.load(os.path.join(BASE_DIR, f'classifier_rf.pkl'))

X_scaled = scaler.transform(X)

y_pred_rf = clf_rf.predict(X_scaled)

df_classification = pd.DataFrame({
    'METER_ID': df_results_m['METER_ID'],
    'HP': y_pred_rf[:, 0],
    'PV': y_pred_rf[:, 1],
    'EV': y_pred_rf[:, 2]
})

df_elec_classified = df_elec.merge(df_classification, on='METER_ID', how='left')

#%% Identified profiles

fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# HP only
df_hp = df_elec_classified[df_elec_classified['HP'] == 1]
mean_hp = df_hp.groupby('DATES')['VOLUME_KWH'].mean()

# PV only
df_pv = df_elec_classified[df_elec_classified['PV'] == 1]
mean_pv = df_pv.groupby('DATES')['VOLUME_KWH'].mean()

# EV only
df_ev = df_elec_classified[df_elec_classified['EV'] == 1]
mean_ev = df_ev.groupby('DATES')['VOLUME_KWH'].mean()

axs[0].plot(mean_hp.index, mean_hp.values, color='tomato', lw=2)
axs[1].plot(mean_pv.index, mean_pv.values, color='gold', lw=2)
axs[2].plot(mean_ev.index, mean_ev.values, color='steelblue', lw=2)

for ax in axs:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

axs[0].set_title('HP only')
axs[1].set_title('PV only')
axs[2].set_title('EV only')

plt.show()

# %%
