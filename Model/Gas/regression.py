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

plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_gas = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Gas/gas_daily.feather'))
df_meteo = pd.read_feather(os.path.join(BASE_DIR, '../../Data/Meteo/meteo_daily.feather'))
df_day = pd.read_feather(os.path.join(BASE_DIR, '../../Data/day.feather'))

#%% Modelling features

MODEL_DEFS = {
    'M01': dict(param_names=['y0', 'alpha', 'Th'], # Linear Self-PV
               smooth=False, lse=False),
    'M2': dict(param_names=['y0', 'alpha', 'Th', 'w'], # Thermal inertia
               smooth=True, lse=False),
    'M3': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma'], # Solar Heat Gain
               smooth=True, lse=False),
    'M4': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma', 'omega'], # Wind chill
               smooth=True, lse=False),
    'M5': dict(param_names=['y0', 'alpha', 'Th', 'w', 'gamma', 'omega', 'k'], # Smooth transition regime
               smooth=True, lse=True)
}

def smooth(x, w, n):
    """
    Applies a moving average smoothing to the input array x with a window size n and a weighting factor w.
    """
    weights = (1 / w) ** np.arange(n + 1)
    weights /= weights.sum()  # Normalise weights

    x_smooth = x.copy()
    conv = np.convolve(x, weights)[:len(x)]
    x_smooth[n:] = conv[n:]

    return x_smooth

def model(x, params, param_names, smooth_flag, lse_flag):
    """
    Compute the predicted consumption from the grid based on the model parameters and input features.
    """
    T, G, W = x
    p = dict(zip(param_names, params))

    # Base component
    y_cst = p['y0']

    # Heating component
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

    y_hp = plc * hdd

    y = y_cst + y_hp

    return y

def objective_function(
        params, x, y,
        prior, prior_scale,
        model_fn
):
    """
    Compute the objective function associated to Maximum A Posteriori (MAP) estimation for the model parameters.
    """
    pred = model_fn(x, params)
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

def get_param_spec(T, G, W, y):
    """
    Set initial guesses, scales, and bounds for the model parameters based on the input features and consumption data.
    """
    dy = np.percentile(y, 95) - np.percentile(y, 5) # Robust range of y

    specs = {
        'y0': (np.percentile(y, 50), dy / 2, (1e-6, None)), # Init, Scale, Bounds
        'alpha': (0, dy / (18 - T.min()), (0, None)),
        'Th': (18, 5, (T.min(), T.max())),
        'gamma': (0, 3 / G.mean(), (0, 1 / G.mean())),
        'omega': (0, dy / (18 - T.min()) * 1 / W.mean(), (0, None)),
        'k':     (1, 10, (1e-6, 10)),
        'w':     (2, 10, (1e-6, 100))
    }

    return specs

def get_x0_xs_bounds(T, G, W, y, param_names):
    """
    Get initial guesses, scales, and bounds for the model parameters based on the input features and consumption data.
    """
    table = get_param_spec(T, G, W, y)
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
        smooth_flag=cfg['smooth'],
        lse_flag=cfg['lse']
    )

    for customer, df_customer in tqdm(df_gas.groupby('EAN_ID'), desc='Customers', unit='customer'):

        y = df_customer['CONS_sum'].values

        if y.max() == 0:
            continue # Skip customers with zero consumption

        y_norm = y / y.max() # Normalise consumption

        x0, xs, bounds = get_x0_xs_bounds(T, G, W, y_norm, cfg['param_names'])

        prior = x0
        prior_scale = xs

        result = minimize(
            objective_function,
            x0=x0,  # Initial guess
            args=(x, y_norm, prior, prior_scale, model_fn),
            bounds=bounds,
            method='L-BFGS-B'
        )

        params_opt = result.x
        loss_opt = result.fun

        y_pred = model_fn(x, params_opt)
        res = y_norm - y_pred

        sse = np.sum(res**2)
        sst = np.sum((y_norm - np.mean(y_norm))**2)
        r2 = 1 - sse / sst
        rmse = np.sqrt(np.mean(res**2))

        # compute AIC and BIC
        k = len(params_opt)
        n = len(y_norm)
        aic = n * np.log(sse / n) + 2 * k
        bic = n * np.log(sse / n) + k * np.log(n)

        opt_dict = dict(zip(cfg['param_names'], np.column_stack((prior, params_opt)))) # Prior and posterior

        acf_res = acf(res, nlags=15)

        results.append({
            'MODEL': model_id,
            'METER_TYPE': df_customer['METER_TYPE'].iloc[0],
            'EAN_ID': customer,
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
            'sigma': np.std(res),
            'acf1': acf_res[1], # Day before
            'acf7': acf_res[7], # Last week
        })

df_results = pd.DataFrame(results)

# Save df_results locally
df_results.to_pickle('regression_results.pkl')

#%% Model comparison

# MLP
df_mlp = pd.read_pickle('mlp_results.pkl')

model_order = ['M01', 'M2', 'M3', 'M4', 'M5']
params = ['y0', 'alpha', 'Th']

fig = plt.figure(figsize=(12, 5))

gs = fig.add_gridspec(
    nrows=1, ncols=2,
    width_ratios=[1, 1],
    wspace=0.5, hspace=0.3
)

gs_params = gs[:, 1].subgridspec(len(params), 1, hspace=0.5)

axm = fig.add_subplot(gs[0, 0])

axp = [fig.add_subplot(gs_params[i, 0]) for i in range(len(params))]

# Metrics

x = np.arange(len(model_order) + 1)
width = 0.35

r2_mean = [df_results[df_results['MODEL'] == model]['R2'].mean() for model in model_order] + [df_mlp['R2'].mean()]
r2_std = [df_results[df_results['MODEL'] == model]['R2'].std() for model in model_order] + [df_mlp['R2'].std()]

rmse_mean = [df_results[df_results['MODEL'] == model]['RMSE'].mean() for model in model_order] + [df_mlp['RMSE'].mean()]
rmse_std = [df_results[df_results['MODEL'] == model]['RMSE'].std() for model in model_order] + [df_mlp['RMSE'].std()]

aic_mean = [df_results[df_results['MODEL'] == model]['AIC'].mean() for model in model_order] + [df_mlp['AIC'].mean()]
bic_mean = [df_results[df_results['MODEL'] == model]['BIC'].mean() for model in model_order] + [df_mlp['BIC'].mean()]

print("AIC mean:", aic_mean)
print("BIC mean:", bic_mean)

models = model_order + ['MLP']

best_aic_model = models[np.argmin(aic_mean)]
best_bic_model = models[np.argmin(bic_mean)]
print("Best AIC model:", best_aic_model)
print("Best BIC model:", best_bic_model)

axm.bar(x - width/2, r2_mean, width=width, yerr=r2_std, capsize=5, alpha=0.6, color='steelblue', error_kw={'ecolor': 'steelblue'})
axm.set_ylabel('R2', color='steelblue')
axm.tick_params(axis='y', colors='steelblue')

axm2 = axm.twinx()
axm2.bar(x + width/2, rmse_mean, width=width, yerr=rmse_std, capsize=5, alpha=0.6, color='tomato', error_kw={'ecolor': 'tomato'})
axm2.set_ylabel('RMSE', color='tomato')
axm2.tick_params(axis='y', colors='tomato')

axm.axvline(x=len(model_order) - 0.5, color='k', linestyle='--', lw=1)

def tight_ylim(ax, values, std):
    low  = min(values) - max(std)
    high = max(values) + max(std)
    margin = 0.05 * (high - low)
    ax.set_ylim(low - margin, high + margin)

tight_ylim(axm, r2_mean, r2_std)
tight_ylim(axm2, rmse_mean, rmse_std)

axm.set_xticks(x + width/2)
axm.set_xticklabels(model_order + ['MLP'])
axm.set_title("Metrics")

# Parameters
for p in params:

    p_mean = [df_results.loc[df_results['MODEL'] == model, p].apply(lambda x: x[1]).mean() for model in model_order]
    p_std = [df_results.loc[df_results['MODEL'] == model, p].apply(lambda x: x[1]).std() for model in model_order]

    axp[params.index(p)].plot(model_order, p_mean, marker='o', color='orange')
    axp[params.index(p)].fill_between(model_order, np.array(p_mean) - np.array(p_std), np.array(p_mean) + np.array(p_std), alpha=0.2, color='orange')

    axp[params.index(p)].set_ylabel(p)

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Regression/metrics_gas_model.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Regression/metrics_gas_model.pdf'), bbox_inches='tight')

plt.show()

#%% Result analysis

# Select model
m = 'M3'
df_res = df_results[df_results['MODEL'] == m]
valid_params = MODEL_DEFS[m]['param_names']

for group, df_group in df_res.groupby('METER_TYPE'):
    print(f"Meter type: {group}")

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

    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Regression/{group}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/Regression/{group}.pdf'), bbox_inches='tight')

    plt.show()
# %%
