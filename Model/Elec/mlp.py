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

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42) # Set random seed for reproducibility

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

#%% Modelling features

class MLP(nn.Module):
    # Define a Multi-Layer Perceptron (MLP) model with a specified number of hidden layers and neurons
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Define the forward pass of the MLP model
        return self.network(x)

def scale(X, Xmin, Xmax):
    """
    Scale the input features to the range [-1, 1] based on the minimum and maximum values.
    """
    return 2 * (X - Xmin) / (Xmax - Xmin) - 1

# Features
T = df_meteo['TEMP_AVG'].values
G = df_meteo['GHI'].values
W = df_meteo['WIND_SPEED_10M'].values

t = df_day['DATE'].values

X = np.column_stack([T, G, W])
X_t = torch.tensor(X, dtype=torch.float32)
X_min, X_max  = X_t.min(0).values, X_t.max(0).values
X_s = scale(X_t, X_min, X_max)

# Set hyperparameters for the MLP model
input_size = X_s.shape[1]
hidden_size = 8
num_hidden_layers = 2

prior_scale = 2 # Set the scale for the prior distribution of the model parameters

n_epochs = 2000
lr = 1e-2

results = [] # Store results for all models and customers

for customer, df_customer in tqdm(df_elec.groupby('EAN_ID'), desc='Customers', unit='customer'):

    y = df_customer['CONS_sum'].values

    if y.max() == 0:
        continue # Skip customers with zero consumption

    y_norm = y / y.max() # Normalise consumption
    y_t = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(42) # Set random seed for reproducibility
    model = MLP(input_size, hidden_size, 1, num_hidden_layers)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    loss_list = [] # Store loss values for each epoch

    for epoch in range(n_epochs):
        pred = model(X_s)
        res = y_t - pred

        # Compute the negative log-likelihood (NLL) and negative log-prior (NLP) for the model for Maximum A Posteriori (MAP) estimation
        n_obs = len(res)
        sse = torch.sum(res ** 2)
        sigma2 = torch.var(res, unbiased=False)
        nll = n_obs / 2 * torch.log(2 * np.pi * sigma2) + sse / (2 * sigma2)

        nlprior = 1 / 2 * sum(torch.sum(param ** 2) for param in model.parameters()) / prior_scale ** 2 # N(0, prior_scale^2) prior
        
        nlposterior = nll + nlprior

        loss = nlposterior

        # Update model parameters using backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_list.append(loss.item())

    with torch.no_grad(): # Evaluate the model on the training data without computing gradients
        y_pred = model(X_s).squeeze().numpy()

    res = y_norm.flatten() - y_pred
    sse = np.sum(res ** 2)
    sst = np.sum((y_norm.flatten() - np.mean(y_norm.flatten())) ** 2)
    r2 = 1 - sse / sst
    rmse = np.sqrt(np.mean(res ** 2))

    # Compute AIC and BIC
    k = sum(p.numel() for p in model.parameters())
    n = len(y_norm)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)

    acf_res = acf(res, nlags=15)

    results.append({
        'SYSTEM': df_customer['SYSTEM'].iloc[0],
        'SYSTEM_LBL': df_customer['SYSTEM_LBL'].iloc[0],
        'EAN_ID': customer,
        'HP': df_customer['HP'].iloc[0],
        'PV': df_customer['PV'].iloc[0],
        'EV': df_customer['EV'].iloc[0],
        'true': y_norm,
        'pred': y_pred,
        'ymax': y.max(),
        'loss_list': loss_list,
        'loss_opt': loss.item(),
        'res': res,
        'acf_res': acf_res,
        'R2': r2,
        'RMSE': rmse,
        'AIC': aic,
        'BIC': bic,
        'n_params': k,
        'params': [p.detach().numpy() for p in model.parameters()],
        'sigma': np.std(res),
        'acf1': acf_res[1], # Day before
        'acf7': acf_res[7], # Last week
    })
    
df_results = pd.DataFrame(results)

# Save df_results locally
df_results.to_pickle('mlp_results.pkl')

# Plot the mean and std of the loss for all customers
fig, ax = plt.subplots(figsize=(8, 5))

loss_mean = np.mean([r['loss_list'] for r in results], axis=0)
loss_std = np.std([r['loss_list'] for r in results], axis=0)

ax.plot(loss_mean, label='Mean Loss')
ax.fill_between(np.arange(len(loss_mean)), loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, label='Std Dev')

ax.set_xlabel('Epoch')
ax.set_ylabel('Negative Log Posterior')
ax.legend()
ax.set_title('Training Loss Evolution')

plt.savefig(os.path.join(BASE_DIR, f'../../Figures/MLP/training_loss_evolution_elec.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, f'../../Figures/MLP/training_loss_evolution_elec.pdf'), bbox_inches='tight')

plt.show()

#%% Result analysis

for group, df_group in df_results.groupby('SYSTEM_LBL'):
    print(f"System type: {group}")

    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1, 1],
        wspace=0.5, hspace=0.4
    )

    ax11 = fig.add_subplot(gs[0, 0])
    ax21 = fig.add_subplot(gs[1, 0])

    gs_ax12 = gs[0, 1].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    ax12 = [fig.add_subplot(gs_ax12[i, j]) for i in range(2) for j in range(2)]

    ax22 = fig.add_subplot(gs[1, 1])

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

    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/MLP/{group}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, f'../../Figures/MLP/{group}.pdf'), bbox_inches='tight')

    plt.show()
#%% Metrics

r2_mean = df_results['R2'].mean()
r2_std = df_results['R2'].std()

rmse_mean = df_results['RMSE'].mean()
rmse_std = df_results['RMSE'].std()

aic_mean = df_results['AIC'].mean()
bic_mean = df_results['BIC'].mean()

# %%
