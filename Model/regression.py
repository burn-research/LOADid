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

#%% Plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('text', usetex=True) # Use LaTeX for rendering text
plt.rc('font', family='serif', size=12)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

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

    ax.legend(title=group, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30); fig.subplots_adjust(hspace=0.35)

    ax.set_xlabel('Date')
    if i == 0:
        ax.set_ylabel(r'Gas Consumption $\left[\frac{m^3}{day}\right]$')

plt.savefig(os.path.join(BASE_DIR, '../Figures/gas_meter_types.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, '../Figures/gas_meter_types.pdf'), bbox_inches='tight')

plt.show()

#%% Bayesian regression
import pymc as pm
import arviz as az
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

T = df_meteo['TEMP_AVG'].values

var = 'CONS_sum'
df_ensemble = df_gas.groupby('DATE').agg(
    mean = (var, 'mean'),
    median = (var, 'median'),
    std = (var, 'std'),
    q25 = (var, lambda x: x.quantile(0.25)),
    q75 = (var, lambda x: x.quantile(0.75))
).reset_index()

y = df_ensemble['mean'].values

# y = y0 + a * max(0, base - T) + noise
with pm.Model() as model:
    # Priors
    y0_mu = np.percentile(y, 10)
    y0 = pm.Gamma('y0', mu=y0_mu, sigma=0.5 * y0_mu)
    a = pm.HalfNormal('a', sigma=y.std() / T.std())
    base = pm.TruncatedNormal('base', mu=15, sigma=3, lower=0, upper=30)
    sigma = pm.HalfNormal('sigma', sigma=0.5 * y.std())

    # Expected value of outcome
    hdd = pm.math.maximum(0, base - T) # pm.math.softplus(base - T, beta=10)
    mu = y0 + a * hdd

    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sample priors before fitting, for comparison
    prior = pm.sample_prior_predictive(samples=2000)

    # Inference
    trace = pm.sample(samples=2000, tune=2000, chains=10, cores=1, target_accept=0.9)
    posterior_pred = pm.sample_posterior_predictive(trace)

# Plot prior vs posterior parameter distributions
from scipy.stats import gamma, halfnorm, truncnorm

param_names = ['y0', 'a', 'base', 'sigma']

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for ax, name in zip(axs.flatten(), param_names):
    # Prior
    prior_samples = prior.prior[name].values.flatten()
    ax.hist(prior_samples, bins=30, density=True, alpha=0.5, color='steelblue')

    x = np.linspace(prior_samples.min(), prior_samples.max(), 100)
    if name == 'y0':
        mu, std = np.percentile(y, 10), 0.5 * np.percentile(y, 10)
        var, alpha, beta = std**2, mu**2 / std**2, mu / std**2
        pdf = gamma.pdf(x, a=alpha, scale=1/beta)
    elif name == 'a':
        std = y.std() / T.std()
        pdf = halfnorm.pdf(x, scale=std)
    elif name == 'base':
        mu, std = 15, 3
        a, b = (0 - mu) / std, (30 - mu) / std
        pdf = truncnorm.pdf(x, a=a, b=b, loc=mu, scale=std)
    elif name == 'sigma':
        std = 0.5 * y.std()
        pdf = halfnorm.pdf(x, scale=std)
    ax.plot(x, pdf, color='steelblue', lw=2, label='Prior')

    # Posterior
    posterior_samples = trace.posterior[name].values.flatten()
    ax.hist(posterior_samples, bins=30, density=True, alpha=0.5, color='orangered', label='Posterior')
    ax.axvline(posterior_samples.mean(), color='orangered', lw=2, ls='--', label='Posterior mean')

    ax.set_title(name)
    ax.legend()

fig.subplots_adjust(hspace=0.35)

plt.show()

# Plot predicted consumption vs temperature with uncertainty band
order = np.argsort(T)
T_sorted, y_sorted = T[order], y[order]

pred_samples = posterior_pred.posterior_predictive['y_obs'].values.reshape(-1, len(y))
pred_sorted = pred_samples[:, order]

pred_mean = pred_sorted.mean(axis=0)
pred_low, pred_high = np.percentile(pred_sorted, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(T_sorted, y_sorted, s=15, color='black', label='Observed')
ax.plot(T_sorted, pred_mean, color='r', lw=2, label='Posterior mean')
ax.fill_between(T_sorted, pred_low, pred_high, color='r', alpha=0.2, label='95\% credible interval')

ax2 = ax.twinx()
ax2.set_yticks([])

base_samples = trace.posterior['base'].values.flatten()
base_mean = base_samples.mean()

ax2.axvline(base_mean, color='orangered', lw=2, ls='--', label='Base temperature')
ax2.hist(trace.posterior['base'].values.flatten(), bins=30, density=True, alpha=0.5, color='orangered', label='Posterior base temperature')

ax.set_xlabel('Average temperature (°C)')
ax.set_ylabel('Gas consumption (kWh/day)')
ax.legend(loc='lower left', frameon=False)
ax2.legend(bbox_to_anchor=(0.4, 1), loc='upper left', facecolor='white', edgecolor='none', framealpha=0.8)
plt.show()

# print the mean y0, a, base, and sigma values
y0_mean = trace.posterior['y0'].values.mean()
a_mean = trace.posterior['a'].values.mean()
base_mean = trace.posterior['base'].values.mean()

# %%
