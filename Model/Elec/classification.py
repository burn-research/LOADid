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
df_day = pd.read_feather(os.path.join(BASE_DIR, '../../Data/day.feather'))

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
T = df_meteo['TEMP_AVG'].values
G = df_meteo['GHI'].values
W = df_meteo['WIND_SPEED_10M'].values

t = df_day['DATE'].values

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

    for customer, df_customer in tqdm(df_elec.groupby('EAN_ID'), desc='Customers', unit='customer'):

        # Impose 1 for classification purposes
        heat_flag = 1 # df_customer['HP'].iloc[0] == 1
        pv_flag = 1 # df_customer['PV'].iloc[0] == 1

        y = df_customer['CONS_sum'].values

        if y.max() == 0:
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

#%% Result analysis

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

for group, df_group in df_results_m.groupby('SYSTEM_LBL'):
    print(f"System type: {group}")

    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1, 1, 1],
        wspace=0.5, hspace=0.4
    )

    gs_params = gs[:, 2].subgridspec(len(valid_params), 1, hspace=0.5)

    ax11 = fig.add_subplot(gs[0, 0])
    ax21 = fig.add_subplot(gs[1, 0])

    gs_ax12 = gs[0, 1].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    ax12 = [fig.add_subplot(gs_ax12[i, j]) for i in range(2) for j in range(2)]

    ax22 = fig.add_subplot(gs[1, 1])

    axs_right = [fig.add_subplot(gs_params[i, 0]) for i in range(len(valid_params))]

    # Residuals
    res_mean = np.mean(np.array(df_group['res'].values.tolist()), axis=0)
    res_quantiles = np.percentile(np.array(df_group['res'].values.tolist()), [25, 75], axis=0)

    # Plot of mean residuals over time
    ax11.plot(t, res_mean, color='k', lw=1)
    ax11.fill_between(t, res_quantiles[0], res_quantiles[1], color='k', alpha=.08, edgecolor='none', label='IQR')
    ax11.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax11.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax11.tick_params(axis='x', rotation=30, labelsize=10)

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

    ax12[2].scatter(T, df_group['true'].values.mean(), color='k', alpha=0.3, s=6, label='Obs')
    ax12[2].scatter(T, df_group['pred'].values.mean(), color='tomato', alpha=0.3, s=6, label='Pred')
    ax12[2].set_xlabel(r'Outside\\Temperature $\left[^{\circ}C\right]$')
    ax12[2].set_ylabel('Mean Normalised\nconsumption')
    ax12[2].legend(frameon=False, fontsize=8)

    ax12[3].scatter(G, df_group['true'].values.mean(), color='k', alpha=0.3, s=6, label='Obs')
    ax12[3].scatter(G, df_group['pred'].values.mean(), color='tomato', alpha=0.3, s=6, label='Pred')
    ax12[3].set_xlabel(r'GHI $\left[\frac{W}{m^2}\right]$')
    ax12[3].set_yticks([])
    ax12[3].legend(frameon=False, fontsize=8)

    # Boxplot of ACF values
    acf_matrix = np.array(df_group['acf_res'].values.tolist())
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
        priors = [v[0] for v in df_group[param]]
        posteriors = [v[1] for v in df_group[param]]

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

    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/{group}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/{group}.pdf'), bbox_inches='tight')

    plt.show()

#%% Features comparison

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

fig = plt.figure(figsize=(18, 12))

gs = fig.add_gridspec(
    nrows=1, ncols=3,
    width_ratios=[1, 1, 1],
    wspace=0.3, hspace=0.3
)

# HP
gs_hp = gs[:, 0].subgridspec(len(valid_params), 1, hspace=0.5)
axs_hp = [fig.add_subplot(gs_hp[i, 0]) for i in range(len(valid_params))]

df_hp = df_results_m[df_results_m['HP'] == 1]; df_no_hp = df_results_m[df_results_m['HP'] == 0]

for i, param in enumerate(valid_params):
    post_hp = [v[1] for v in df_hp[param]]; post_no_hp = [v[1] for v in df_no_hp[param]]
    bp_hp = axs_hp[i].boxplot(
        post_hp, orientation='horizontal', positions = [1], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green'),
    )
    bp_no_hp = axs_hp[i].boxplot(
        post_no_hp, orientation='horizontal', positions = [2], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightcoral'),
        medianprops=dict(color='red'),
    )

    axs_hp[i].set_ylabel(param)
    axs_hp[i].set_yticks([])

    if i == 0:
        legend_handles = [
            Patch(facecolor='lightgreen', edgecolor='k', label='HP'),
            Patch(facecolor='lightcoral', edgecolor='k', label='No HP')
        ]
        axs_hp[i].legend(handles=legend_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

# PV
gs_pv = gs[:, 1].subgridspec(len(valid_params), 1, hspace=0.5)
axs_pv = [fig.add_subplot(gs_pv[i, 0]) for i in range(len(valid_params))]

df_pv = df_results_m[df_results_m['PV'] == 1]; df_no_pv = df_results_m[df_results_m['PV'] == 0]

for i, param in enumerate(valid_params):
    post_pv = [v[1] for v in df_pv[param]]; post_no_pv = [v[1] for v in df_no_pv[param]]
    bp_pv = axs_pv[i].boxplot(
        post_pv, orientation='horizontal', positions = [1], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green'),
    )
    bp_no_pv = axs_pv[i].boxplot(
        post_no_pv, orientation='horizontal', positions = [2], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightcoral'),
        medianprops=dict(color='red'),
    )

    axs_pv[i].set_ylabel(param)
    axs_pv[i].set_yticks([])

    if i == 0:
        legend_handles = [
            Patch(facecolor='lightgreen', edgecolor='k', label='PV'),
            Patch(facecolor='lightcoral', edgecolor='k', label='No PV')
        ]
        axs_pv[i].legend(handles=legend_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

# EV
gs_ev = gs[:, 2].subgridspec(len(valid_params), 1, hspace=0.5)
axs_ev = [fig.add_subplot(gs_ev[i, 0]) for i in range(len(valid_params))]

df_ev = df_results_m[df_results_m['EV'] == 1]; df_no_ev = df_results_m[df_results_m['EV'] == 0]

for i, param in enumerate(valid_params):
    post_ev = [v[1] for v in df_ev[param]]; post_no_ev = [v[1] for v in df_no_ev[param]]
    bp_ev = axs_ev[i].boxplot(
        post_ev, orientation='horizontal', positions = [1], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green'),
    )
    bp_no_ev = axs_ev[i].boxplot(
        post_no_ev, orientation='horizontal', positions = [2], widths=0.5,
        showfliers=False, flierprops=dict(marker='.', markerfacecolor='k', markeredgecolor='none', alpha=0.1),
        patch_artist=True, boxprops=dict(facecolor='lightcoral'),
        medianprops=dict(color='red'),
    )

    axs_ev[i].set_ylabel(param)
    axs_ev[i].set_yticks([])

    if i == 0:
        legend_handles = [
            Patch(facecolor='lightgreen', edgecolor='k', label='EV'),
            Patch(facecolor='lightcoral', edgecolor='k', label='No EV')
        ]
        axs_ev[i].legend(handles=legend_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/features_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/features_comparison.pdf'), bbox_inches='tight')

plt.show()

#%% Features distribution

# Select model
m = 'M3'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
plt.subplots_adjust(wspace=0.5)

# HP

alpha_values = [v[1] for v in df_results_m['alpha']]
hp_values = [v for v in df_results_m['HP']]

bins = np.linspace(min(alpha_values), max(alpha_values), 50)

counts, edges = np.histogram(alpha_values, bins=bins) # Number of samples in each bin
ones, _ = np.histogram(np.array(alpha_values)[np.array(hp_values) == 1], bins=bins) # Number of samples with HP=1 in each bin
density_1 = np.divide(ones, counts, out=np.zeros_like(ones, dtype=float), where=counts > 0) # Fraction of HP=1 in each bin

cmap = plt.cm.Reds
norm = plt.Normalize(vmin=0, vmax=1)

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[0].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color=cmap(norm(density_1)),
    edgecolor='none',
    alpha=0.7
)

# KDE
sns.kdeplot(alpha_values, color='k', lw=1.5, ax=axs[0], clip=(min(alpha_values), max(alpha_values)))

# Find modes using KDE
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

kde = gaussian_kde(alpha_values)

x_kde = np.linspace(min(alpha_values), max(alpha_values), 2000)
y_kde = kde(x_kde)

# Find local maxima
peaks, _ = find_peaks(y_kde)

# Mode locations (first two)
modes = x_kde[peaks][:2]

print(modes[1] / modes[0])  # Ratio of the two modes

# Plot vertical lines
for mode in modes:
    axs[0].axvline(mode, color='k', linestyle='--', linewidth=1.2)

cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:, -1] = 0.7
cmap_alpha = plt.matplotlib.colors.ListedColormap(cmap_alpha)

sm = plt.cm.ScalarMappable(cmap=cmap_alpha, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs[0])
cbar.set_label('Fraction of HP')

axs[0].set_xlabel(r'$\alpha$', fontsize=16)

# PV

rho_values = [v[1] for v in df_results_m['rho']]
pv_values = [v for v in df_results_m['PV']]

bins = np.linspace(min(rho_values), max(rho_values), 50)

counts, edges = np.histogram(rho_values, bins=bins) # Number of samples in each bin
ones, _ = np.histogram(np.array(rho_values)[np.array(pv_values) == 1], bins=bins) # Number of samples with PV=1 in each bin
density_1 = np.divide(ones, counts, out=np.zeros_like(ones, dtype=float), where=counts > 0) # Fraction of PV=1 in each bin

cmap = plt.cm.YlOrBr
norm = plt.Normalize(vmin=0, vmax=1)

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[1].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color=cmap(norm(density_1)),
    edgecolor='none',
    alpha=0.7
)

# KDE
sns.kdeplot(rho_values, color='k', lw=1.5, ax=axs[1], clip=(min(rho_values), max(rho_values)))

cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:, -1] = 0.7
cmap_alpha = plt.matplotlib.colors.ListedColormap(cmap_alpha)

sm = plt.cm.ScalarMappable(cmap=cmap_alpha, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs[1])
cbar.set_label('Fraction of PV')

axs[1].set_xlabel(r'$\rho$', fontsize=16)

# EV

ymax_values = [v for v in df_results_m['ymax']]
ev_values = [v for v in df_results_m['EV']]

bins = np.linspace(min(ymax_values), max(ymax_values), 50)

counts, edges = np.histogram(ymax_values, bins=bins) # Number of samples in each bin
ones, _ = np.histogram(np.array(ymax_values)[np.array(ev_values) == 1], bins=bins) # Number of samples with EV=1 in each bin
density_1 = np.divide(ones, counts, out=np.zeros_like(ones, dtype=float), where=counts > 0) # Fraction of EV=1 in each bin

cmap = plt.cm.Blues
norm = plt.Normalize(vmin=0, vmax=1)

bin_width = edges[1] - edges[0]
density = counts / (counts.sum() * bin_width)  # Density for histogram

# Histogram
axs[2].bar(
    (edges[:-1] + edges[1:]) / 2,
    density,
    width=edges[1:] - edges[:-1],
    color=cmap(norm(density_1)),
    edgecolor='none',
    alpha=0.7
)

# KDE
sns.kdeplot(ymax_values, color='k', lw=1.5, ax=axs[2], clip=(min(ymax_values), max(ymax_values)))

cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:, -1] = 0.7
cmap_alpha = plt.matplotlib.colors.ListedColormap(cmap_alpha)

sm = plt.cm.ScalarMappable(cmap=cmap_alpha, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs[2])
cbar.set_label('Fraction of EV')

axs[2].set_xlabel(r'$y_{max}$', fontsize=16)

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/features_distribution.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/features_distribution.pdf'), bbox_inches='tight')

plt.show()

#%% Classification on features

from sklearn.preprocessing import StandardScaler # RF is scale-invariant
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

# Select model
m = 'M5'
df_results_m = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

params = valid_params + ['ymax', 'acf1', 'acf7']  # Features for classification

# Multilabel classification for HP, PV and EV
X = df_results_m[params].map(lambda x: x if isinstance(x, (int, float)) else x[1]) # Use posterior values when relevant
y = df_results_m[['HP', 'PV', 'EV']]

# Stack X and y in D
D = pd.concat([X, y], axis=1)

# Train test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, D.index, stratify=y, test_size=0.2, random_state=42
)
D['TRAIN_TEST'] = ['Train' if i in idx_train else 'Test' for i in D.index]

# Scale Xtrain and use the same scaler for Xtest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Multilabel classifier - independent labels

# Random Forest Classifier
clf_rf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
)
clf_rf.fit(X_train_scaled, y_train)

# Save the scaler and classifier for Brussels Case
joblib.dump(scaler, os.path.join(BASE_DIR, f'../../Brussels/scaler.pkl'))
joblib.dump(clf_rf, os.path.join(BASE_DIR, f'../../Brussels/classifier_rf.pkl'))

# Gaussian Process Classifier
kernel = 1.0 * RBF(length_scale=1.0)
clf_gp = MultiOutputClassifier(
    GaussianProcessClassifier(
        kernel = kernel,
        random_state=42,
        n_jobs=-1
    )
)
clf_gp.fit(X_train_scaled, y_train)

# Predict for all customers
y_pred_rf = clf_rf.predict(X_scaled)
y_pred_gp = clf_gp.predict(X_scaled)

y_pred_rf_proba = clf_rf.predict_proba(X_scaled)
y_pred_gp_proba = clf_gp.predict_proba(X_scaled) # [[(P(HP=0), P(HP=1)), (P(PV=0), P(PV=1)), (P(EV=0), P(EV=1))]]

# Attach predictions to D and df_results_m
D['HP_RF'] = y_pred_rf[:, 0]; D['PV_RF'] = y_pred_rf[:, 1]; D['EV_RF'] = y_pred_rf[:, 2]
df_results_m['HP_RF'] = y_pred_rf[:, 0]; df_results_m['PV_RF'] = y_pred_rf[:, 1]; df_results_m['EV_RF'] = y_pred_rf[:, 2]

D['HP_GP'] = y_pred_gp[:, 0]; D['PV_GP'] = y_pred_gp[:, 1]; D['EV_GP'] = y_pred_gp[:, 2]
df_results_m['HP_GP'] = y_pred_gp[:, 0]; df_results_m['PV_GP'] = y_pred_gp[:, 1]; df_results_m['EV_GP'] = y_pred_gp[:, 2]

# Metrics
print('Classification Report - Random Forest')
print(classification_report(y_test, clf_rf.predict(X_test_scaled), target_names=y.columns.astype(str).tolist())) # RF
print('Classification Report - Gaussian Process')
print(classification_report(y_test, clf_gp.predict(X_test_scaled), target_names=y.columns.astype(str).tolist())) # GP

# Confusion matrix - RF
mcm = multilabel_confusion_matrix(y_test, clf_rf.predict(X_test_scaled))

fig, axes = plt.subplots(1, len(y.columns), figsize=(5 * len(y.columns), 4))
fig.suptitle('Random Forest Classifier', fontsize=16)
for ax, label, cm in zip(axes, y.columns, mcm):
    cmap = {'HP': 'Reds', 'PV': 'YlOrBr', 'EV': 'Blues'}[label]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f'Confusion Matrix for {label}')
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/CM_RF.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/CM_RF.pdf'), bbox_inches='tight')

plt.show()

# Confusion matrix - GP
mcm = multilabel_confusion_matrix(y_test, clf_gp.predict(X_test_scaled))

fig, axes = plt.subplots(1, len(y.columns), figsize=(5 * len(y.columns), 4))
fig.suptitle('Gaussian Process Classifier', fontsize=16)
for ax, label, cm in zip(axes, y.columns, mcm):
    cmap = {'HP': 'Reds', 'PV': 'YlOrBr', 'EV': 'Blues'}[label]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f'Confusion Matrix for {label}')
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/CM_GP.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/CM_GP.pdf'), bbox_inches='tight')

plt.show()

# Feature importance
importances_per_label = pd.DataFrame(
    {label: est.feature_importances_ for label, est in zip(y.columns, clf_rf.estimators_)},
    index=params
)

fig, ax = plt.subplots(figsize=(9, 5))

importances_per_label.plot.barh(ax=ax, color=['tomato', 'gold', 'steelblue'])
ax.set_xlabel('Mean decrease in impurity')
ax.set_title('Feature importance by label')
ax.invert_yaxis()
ax.legend(title='Label')
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/feature_importance.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/feature_importance.pdf'), bbox_inches='tight')

plt.show()

#%% Best tree

from sklearn.tree import plot_tree

# Plot the best tree for each label
best_trees = {}

for label, est in tqdm(zip(y.columns, clf_rf.estimators_), desc='Best trees', total=len(y.columns)):
    scores = [
        f1_score(y_test[label], tree.predict(X_test_scaled))
        for tree in est.estimators_
    ]
    best_idx = int(np.argmax(scores))
    best_trees[label] = {
        'tree': est.estimators_[best_idx],
        'f1_score': scores[best_idx],
        'depth': est.estimators_[best_idx].get_depth(),
        'n_leaves': est.estimators_[best_idx].get_n_leaves()
    }

plt.rcParams['text.usetex'] = False # Disable LaTeX rendering for tree visualisation

fig, axs = plt.subplots(1, len(y.columns), figsize=(6 * len(y.columns), 5))
for ax, (label, info) in zip(axs, best_trees.items()):
    plot_tree(
        info['tree'],
        feature_names=params,
        class_names=['0', '1'],
        filled=True,
        ax=ax,
        max_depth=2, # Limit depth for better visualisation
        fontsize=8
    )
    ax.set_title(f'Best Tree for {label}\nF1 Score: {info["f1_score"]:.2f}, Depth: {info["depth"]}, Leaves: {info["n_leaves"]}')
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/best_tree.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/best_tree.pdf'), bbox_inches='tight')

plt.show()

plt.rcParams['text.usetex'] = True # Restore LaTeX rendering

#%% Test-set ratio analysis

# 2400 customers (8 * 300)
training_n_customers = np.array([
    8, 16, 32, 64, 128, 256, 512, 1024, 2048
])
train_sizes = training_n_customers / (8 * 300)
test_sizes = 1 - train_sizes

random_states = range(20)

f1_out_scores = []

for test_size in tqdm(test_sizes, desc='Test sizes', unit='size'):

    f1_out_repeats = []

    for random_state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=random_state
        )
        clf_rf.fit(X_train, y_train)

        f1_out = f1_score(y_test, clf_rf.predict(X_test), average=None)  # Get F1 score for each label

        f1_out_repeats.append(f1_out)

    f1_out_repeats = np.asarray(f1_out_repeats)

    f1_out_scores.append({
        'test_size': test_size,
        'HP_mean': f1_out_repeats[:, 0].mean(),
        'HP_std': f1_out_repeats[:, 0].std(),
        'PV_mean': f1_out_repeats[:, 1].mean(),
        'PV_std': f1_out_repeats[:, 1].std(),
        'EV_mean': f1_out_repeats[:, 2].mean(),
        'EV_std': f1_out_repeats[:, 2].std()
    })

f1_out_df = pd.DataFrame(f1_out_scores)

# Plot F1 score vs number of training customers

fig, ax = plt.subplots(figsize=(6, 5))

for label in y.columns:
    color = {'HP': 'tomato', 'PV': 'gold', 'EV': 'steelblue'}[label]
    mean_out = f1_out_df[f'{label}_mean']
    std_out = f1_out_df[f'{label}_std']

    ax.plot(training_n_customers, mean_out, label=label, color=color)
    ax.fill_between(training_n_customers, mean_out - std_out, mean_out + std_out, color=color, alpha=0.2)

std_patches = [
    Patch(color='tomato', alpha=0.2, label='HP std'),
    Patch(color='gold', alpha=0.2, label='PV std'),
    Patch(color='steelblue', alpha=0.2, label='EV std')
]

ax.legend(
    handles=ax.get_legend_handles_labels()[0] + std_patches,
    title='Label',
    ncol=2,
    columnspacing=1.0,
    handletextpad=0.5,
    labelspacing=0.4,
)

ax.set_xscale('log')
# ax.set_ylim(0, 1)
ax.set_xlabel('Number of training customers')
ax.set_ylabel('F1 Score')
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/train_sizes.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Classification/train_sizes.pdf'), bbox_inches='tight')

plt.show()

#%% TN - FP - FN - TP analysis

fp_hp = df_results_m[(df_results_m['HP'] == 0) & (df_results_m['HP_RF'] == 1)] # Mainly EV customers (96% of them are EV customers)
fn_hp = df_results_m[(df_results_m['HP'] == 1) & (df_results_m['HP_RF'] == 0)] # Half PV customers (48% of them are PV customers)

fp_pv = df_results_m[(df_results_m['PV'] == 0) & (df_results_m['PV_RF'] == 1)] # Mainly HP customers (71% of them are HP customers)
fn_pv = df_results_m[(df_results_m['PV'] == 1) & (df_results_m['PV_RF'] == 0)] # Mainly EV customers (70% of them are EV customers)

fp_ev = df_results_m[(df_results_m['EV'] == 0) & (df_results_m['EV_RF'] == 1)] # Mainly HP customers (82% of them are HP customers)
fn_ev = df_results_m[(df_results_m['EV'] == 1) & (df_results_m['EV_RF'] == 0)] # Half PV customers (52% of them are PV customers)

# %%
