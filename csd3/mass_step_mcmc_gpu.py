#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import numpy as np
import pandas as pd

import torch
import pickle
from tqdm import tqdm
from affine import sample

# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print('device =', device)


# %%


path = '/home/jjg65/rds/hpc-work/ztf_meds/'

HR = np.load(path + 'HR.npy')
HR_err = np.load(path + 'HR_err.npy')
log10M_remain_med = np.load(path + 'log10M_remain_med.npy')
log10SFR_med = np.load(path + 'log10SFR_med.npy') 
log10sSFR_med = np.load(path + 'log10sSFR_med.npy') 
mcmc_dust2_med = np.load(path + 'mcmc_dust2_med.npy')
mcmc_dust1_med = np.load(path + 'mcmc_dust1_med.npy')
mw_age_med = np.load(path + 'mw_age_med.npy')
mcmc_metal_med = np.load(path + 'mcmc_metal_med.npy')
log10M_remain_chains = torch.load(path + 'log10M_remain_chains_nan.pt')

# %%


cond_mass = log10M_remain_med < 10   # note (how does this change if fitting for threshold??) 
cond_ssfr = log10sSFR_med < -11
cond_sfr = log10SFR_med < -1 
cond_dust = mcmc_dust2_med < 0.3
cond_metal = mcmc_metal_med < -0.5 


# %%


# dictionary of conditions
conds = {
    'cond_mass': cond_mass,
    '_cond_mass': ~cond_mass,
    'cond_ssfr': cond_ssfr,
    '_cond_ssfr': ~cond_ssfr,
    'cond_sfr': cond_sfr, 
    '_cond_sfr': ~cond_sfr, 
    'cond_dust': cond_dust,
    '_cond_dust': ~cond_dust,
    'cond_metal': cond_metal,
    '_cond_metal': ~cond_metal
}


# %%


def inputs_by_cond(cond):
    mass_chains = log10M_remain_chains[cond]
    mass = torch.tensor(log10M_remain_med[cond])
    hr = torch.tensor(HR[cond])
    hr_err = torch.tensor(HR_err[cond])
    
    return {'mass_chains': mass_chains, 
            'mass': mass, 
            'hr': hr, 
            'hr_err': hr_err}


# %%


type, fixed = 'fit_M_thresh', False


# %%


# INPUT CONDITION HERE
cond_name = '_cond_ssfr'
cond = conds[cond_name]


# %%


mass_chains = inputs_by_cond(cond)['mass_chains'].to(device)
mass = inputs_by_cond(cond)['mass'].to(device)
hr = inputs_by_cond(cond)['hr'].to(device)
hr_err = inputs_by_cond(cond)['hr_err'].to(device)


# %%


def calculate_p_Im_vals(mass_chains, M_thresh=10): 
    """
    Inputs: mass_chains = mcmc mass chains for selected galaxies (i.e., populations) whose mass we care about, 
                          shape ([n_gal, 511488])
    Outputs: (p_Im_0, p_Im_1) for (narrow, fixed) ordering
    """
    p_Im_1 = torch.sum(mass_chains > M_thresh, dim=1) / mass_chains.shape[1]
    p_Im_0 = 1 - p_Im_1

    return p_Im_0, p_Im_1


# %%


def calculate_p_Im_1(mass_chains, M_thresh=10): 
    p_Im_1 = torch.sum(mass_chains > M_thresh, dim=1) / mass_chains.shape[1]
    return p_Im_1


# %%


def highstep(mass, a): 
    return torch.ones_like(mass) * a


# %%


def lowstep(mass, a, gamma): 
    return torch.ones_like(mass) * (a - gamma)


# %%


def log_likelihood(theta, hr, hr_err, mass, fixed=fixed):
    """Inputs: 
    - theta (N_walkers, n_params)
    - broadcasted data points hr, hr_err, mass (N_walkers, N_gal) 
    """
    gamma = theta[:, 0][:, None].to(device)     # (N_walkers, 1)   (one value for all galaxies fit)
    a = theta[:, 1][:, None].to(device)
    sigma_int = theta[:, 2][:, None].to(device)
    if not fixed: 
        M_thresh = theta[:, 3].to(device)       # (N_walkers) for calculate_p_Im_vals (we have in loop)
    print('gamma:', gamma.shape)
    print('M_thresh:', M_thresh.shape)    # N_walkers

    print('hr:', hr.shape)
    # ensure broadcastable
    if hr.ndim == 1:
        hr = hr[None, :]          # (N_walkers, N_gal)
        hr_err = hr_err[None, :]
        mass = mass[None, :]     
        
    # input mass chains into here based on selected mass 
    if fixed: 
        M_thresh_val = 10        
        p_Ims = calculate_p_Im_vals(mass_chains, M_thresh=M_thresh_val)  # mass_chains is a global variable 
        p_Im_0, p_Im_1 = p_Ims[0], p_Ims[1]
        p_Im_0, p_Im_1 = p_Im_0.broadcast_to(theta.shape[0], -1), p_Im_1.broadcast_to(theta.shape[0], -1)

    if not fixed: 
        p_Im_1 = torch.zeros(theta.shape[0], mass_chains.shape[0]).to(device)   # N_walkers, N_gal
        for i in range(theta.shape[0]):   # 256
            ind_p_Im_1 = calculate_p_Im_1(mass_chains, M_thresh=M_thresh[i]).to(device)
            p_Im_1[i] = ind_p_Im_1
        p_Im_0 = (1 - p_Im_1).to(device)
        
    if p_Im_0.ndim == 1:
        p_Im_0 = p_Im_0[None, :]  
        p_Im_1 = p_Im_1[None, :]
    print('p_Im_0:', p_Im_0.shape)    # (N_walkers, N_gal) -> this is the same for the fixed M_thresh case 
    print(p_Im_0)

    model0 = highstep(mass, a).to(device)             # mass is (N_walkers, N_gal) shape; note don't need the m_thresh
    model1 = lowstep(mass, a, gamma).to(device)
    if model0.ndim == 1: 
        model0 = model0[None, :]   # (N_walkers, N_gal)?
        model1 = model1[None, :]
        
    print('model0:', model0.shape)   # (1, N_gal)
    sigma2 = hr_err**2 + sigma_int**2   # (N_walkers, N_gal)
    print('sigma2:', sigma2.shape)
    
    # fitting sig_int needs to include the normalization term for gaussian 
    norm = (1 / torch.sqrt(2 * torch.pi * sigma2)).to(device)   # (N_walkers, N_gal)
    l0 = p_Im_0 * norm * torch.exp(-0.5 * (hr - model0)**2 / sigma2)   
    l1 = p_Im_1 * norm * torch.exp(-0.5 * (hr - model1)**2 / sigma2)
    print('l0:', l0.shape)
    print('l1:', l1.shape)

    ll = (torch.sum(torch.log(l0 + l1), dim=1)).to(device)    # fine to sum over all walkers??
    print('ll:', ll.shape) 

    return ll   # shape (N_walkers)       # shape 0 


# %%


def log_prior(theta, fixed=fixed):    
    gamma = theta[:, 0]
    a = theta[:, 1]
    sigma_int = theta[:, 2]
    if not fixed: 
        M_thresh = theta[:, 3]

    lp = torch.zeros(len(gamma)).to(device)

    bad = (
        (gamma < -0.5) | (gamma > 0.5) |
        (a < -1) | (a > 1) |
        (sigma_int < 0.0) | (sigma_int > 1)
    )

    lp[bad] = -torch.inf
    
    return lp 


# %%


def log_prob(theta, hr, hr_err, mass, fixed=fixed):
    print('theta:', theta.shape)
    lp = log_prior(theta, fixed=fixed).to(device)
#     if not torch.isfinite(lp):
#         return -torch.inf
    print('lp:', lp.shape)
    ll = log_likelihood(theta, hr, hr_err, mass, fixed=fixed).to(device)
    print('ll:', ll.shape)
    return lp + ll


# %%


def init(hr, hr_err, fixed=fixed): 
    hr_invvar_weightmean = (torch.sum(hr / hr_err**2) / torch.sum(1 / hr_err**2)).to(device)
    gamma0 = 2 * hr_invvar_weightmean
    a0 = gamma0 / 2
    sig_int0 = torch.std(hr).to(device)
    if not fixed: 
        M_thresh0 = 10 
        return gamma0, a0, sig_int0, M_thresh0
    else: 
        return gamma0, a0, sig_int0


# %%


# run mcmc 
def run_mcmc(hr, hr_err, mass, n_params=4, n_walkers=256, n_steps=500, scale=0.05, fixed=False): 
    """Change n_params based on model: simple step = (gamma, a, sig_int)
    """
    print('fixed:', fixed)
    p0 = torch.tensor(init(hr, hr_err, fixed=fixed)).to(device)    # gamma, a, sigma_int

    # two ensembles of walkers, which we'll start in a tight ball around the MAP estimate
    walkers1 = ((torch.randn(n_walkers, n_params)*scale).to(device) + p0).to(device)
    walkers2 = ((torch.randn(n_walkers, n_params)*scale).to(device) + p0).to(device)

    # input broadcasting here for all the walkers 
    hr_broadcast = (hr.broadcast_to(n_walkers, -1)).to(device)
    hr_err_broadcast = (hr_err.broadcast_to(n_walkers, -1)).to(device)
    mass_broadcast = (mass.broadcast_to(n_walkers, -1)).to(device)

    log_prob_func = lambda theta: log_prob(theta, hr_broadcast, hr_err_broadcast, mass_broadcast, fixed=fixed)
#     log_prob_func = lambda theta: log_prob(theta, hr, hr_err, mass)
        
    with torch.no_grad():
        mcmc_theta = sample(log_prob_func, n_params, n_walkers, n_steps, walkers1, walkers2, progress=True, save_lp=False)

    return mcmc_theta


# %%


mcmc_theta = run_mcmc(hr, hr_err, mass, n_params=4, n_walkers=256, n_steps=500, scale=0.05, fixed=False)


# %%
mcmc_theta_half2 = mcmc_theta.detach().numpy()[mcmc_theta.shape[0] // 2:].reshape(-1, 3)
# gamma_fit = mcmc_theta_half2[:, 0]
# a_fit = mcmc_theta_half2[:, 1]
# sig_int_fit = mcmc_theta_half2[:, 2]
np.save(f'{type}_{cond_name}.npy', mcmc_theta_half2)


# %%





# %%





#

