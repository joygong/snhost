#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sncosmo
import astropy
from astropy.cosmology.realizations import Planck18
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import sklearn
import scipy.stats

import torch
import pickle
import corner
from tqdm import tqdm
from affine import sample
import corner
import h5py
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LogNorm


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
fixed = False

# %%
base = '/home/jjg65/rds/hpc-work/'
dir = 'ztf_sn/'
# base = './'
# dir = ''

ztf_snia_vollim = pd.read_csv(base + dir + 'ztf_snia_vollim.csv')
dat = pd.read_csv(base + dir + 'blast_global_data_merged_cuts_SNR_score_z_unext.csv')
dat = dat[dat['ebv_MW'] > 0]       # this is where all fluxes are 0   # it is possible to have the mv_ebv value from the ztf_snia, but regardless there is no photometry :/ 
gal_vollim = dat[(dat['redshift'] > 0.023) & (dat['redshift'] < 0.06)]

cond_narrow = (gal_vollim['redshift_err'] >= 0.001).values
cond_fixed = (gal_vollim['redshift_err'] < 0.001).values

redshift = gal_vollim['redshift'].values
redshift_narrow = gal_vollim[gal_vollim['redshift_err'] >= 0.001]['redshift'].values
redshift_fixed = gal_vollim[gal_vollim['redshift_err'] < 0.001]['redshift'].values
z_cat = np.concatenate((redshift_narrow, redshift_fixed))
z_cat_err = np.concatenate((gal_vollim['redshift_err'][cond_narrow].values,
                           gal_vollim['redshift_err'][cond_fixed].values))


# %%


# peak mag in b band before standardization
cons = sncosmo.Model(source=sncosmo.get_source('salt2', version='T21')).source_peakmag("bessellb", magsys="ab")
mB = -2.5 * np.log10(ztf_snia_vollim['x0']) + cons


# %%


mB_narrow = mB[(gal_vollim['redshift_err'] >= 0.001).values].values
mB_fixed = mB[(gal_vollim['redshift_err'] < 0.001).values].values
mB_cat = np.concatenate((mB_narrow, mB_fixed))

x0_narrow = ztf_snia_vollim['x0'][(gal_vollim['redshift_err'] >= 0.001).values].values
x0_fixed = ztf_snia_vollim['x0'][(gal_vollim['redshift_err'] < 0.001).values].values
x0_cat = np.concatenate((x0_narrow, x0_fixed))

x0_cat_err = np.concatenate((ztf_snia_vollim['x0_err'][(gal_vollim['redshift_err'] >= 0.001).values].values, 
                            ztf_snia_vollim['x0_err'][(gal_vollim['redshift_err'] < 0.001).values].values))

x1_narrow = ztf_snia_vollim['x1'][(gal_vollim['redshift_err'] >= 0.001).values].values
x1_fixed = ztf_snia_vollim['x1'][(gal_vollim['redshift_err'] < 0.001).values].values
x1_cat = np.concatenate((x1_narrow, x1_fixed))

x1_cat_err = np.concatenate((ztf_snia_vollim['x1_err'][(gal_vollim['redshift_err'] >= 0.001).values].values, 
                ztf_snia_vollim['x1_err'][(gal_vollim['redshift_err'] < 0.001).values].values))

c_narrow = ztf_snia_vollim['c'][(gal_vollim['redshift_err'] >= 0.001).values].values
c_fixed = ztf_snia_vollim['c'][(gal_vollim['redshift_err'] < 0.001).values].values
c_cat = np.concatenate((c_narrow, c_fixed))

c_cat_err = np.concatenate((ztf_snia_vollim['c_err'][(gal_vollim['redshift_err'] >= 0.001).values].values, 
                           ztf_snia_vollim['c_err'][(gal_vollim['redshift_err'] < 0.001).values].values))


# %%


sig_x0c = np.concatenate((ztf_snia_vollim['cov_x0_c'][cond_narrow], 
                          ztf_snia_vollim['cov_x0_c'][cond_fixed]))

sig_x0x1 = np.concatenate((ztf_snia_vollim['cov_x0_x1'][cond_narrow], 
                          ztf_snia_vollim['cov_x0_x1'][cond_fixed]))

sig_x1c = np.concatenate((ztf_snia_vollim['cov_x1_c'][cond_narrow].values, 
                ztf_snia_vollim['cov_x1_c'][cond_fixed].values))


# %%


# use planck instead bc gives more reasonable values for HR
cosmo = Planck18
z_cat = torch.tensor(np.concatenate((redshift_narrow, redshift_fixed)))
mu_cat = cosmo.distmod(z_cat).value



# %%


# log10M_remain_med = np.load('log10M_remain_med.npy')
# log10SFR_med = np.load('log10SFR_med.npy') 
dir = 'ztf_meds/'
# dir = ''
log10sSFR_med = np.load(os.path.join(base, dir, 'log10sSFR_med.npy'))
mcmc_dust2_med = np.load(os.path.join(base, dir, 'mcmc_dust2_med.npy'))
# mcmc_dust1_med = np.load('mcmc_dust1_med.npy')
# mw_age_med = np.load('mw_age_med.npy')
# mcmc_metal_med = np.load('mcmc_metal_med.npy')


# ## Joint Parameter Fitting ##

# %%


## mass indicator calculations (for each galaxy, get two weights)
dir = 'ztf_meds/'
# dir = ''

mass_chains = torch.load(base + dir + 'log10M_remain_chains_sorted.pt')
mass_mins = torch.load(base + dir + 'mass_mins.pt')
mass_maxs = torch.load(base + dir + 'mass_mins.pt')



# %%


# dictionary of conditions
# conds = {
#     'cond_mass': cond_mass,
#     '_cond_mass': ~cond_mass,
#     'cond_ssfr': cond_ssfr,
#     '_cond_ssfr': ~cond_ssfr,
#     'cond_sfr': cond_sfr, 
#     '_cond_sfr': ~cond_sfr, 
#     'cond_dust': cond_dust,
#     '_cond_dust': ~cond_dust,
#     'cond_metal': cond_metal,
#     '_cond_metal': ~cond_metal
# }

# def inputs_by_cond(cond):
#     mass_chains = log10M_remain_chains[cond]
#     mass = torch.tensor(log10M_remain_med[cond])
#     hr = torch.tensor(HR[cond])
#     hr_err = torch.tensor(HR_err[cond])
    
#     return {'mass_chains': mass_chains, 
#             'mass': mass, 
#             'hr': hr, 
#             'hr_err': hr_err}

# type, fixed = 'fit_M_thresh', False 
# # INPUT CONDITION HERE
# cond_name = 'cond_ssfr'
# cond = conds[cond_name]

# mass_chains = inputs_by_cond(cond)['mass_chains']
# mass = inputs_by_cond(cond)['mass']
# hr = inputs_by_cond(cond)['hr']
# hr_err = inputs_by_cond(cond)['hr_err']

## this is not necessary 
# def step(mass, a, gamma, M_thresh=10): 
#     """note this is fitting HR vs. mass, in which case:
#         - lower mass = higher HR
#         - higher mass = lower HR """
#     return np.where(mass < M_thresh, a, a - gamma)

# def highstep(mass, a): 
#     return torch.ones_like(mass) * a

# def lowstep(mass, a, gamma): 
#     return torch.ones_like(mass) * (a - gamma)


# %%


def tripp(mB, x1, c, M_B, alpha, beta, gamma): 
    """mB, x1, c are SN-specific values, ordered (narrow, fixed)
    M_B, alpha, beta, gamma are sample-specific constants (+gamma/2 for low-mass; -gamma/2 for high-mass) 
    assume \delta_bias = 0; delta_host = gamma """
    return mB + alpha*x1 - beta*c - M_B + gamma 
    
def calculate_sig_mB(x0=x0_cat, sig_x0=x0_cat_err): 
    return (-2.5 / (x0*np.log(10)))**2 * sig_x0**2

def calculate_sig_mBc(x0=x0_cat, sig_x0c=sig_x0c):
    return (-2.5 / (x0*np.log(10))) * sig_x0c 

def calculate_sig_mBx1(x0=x0_cat, sig_x0x1=sig_x0x1):
    return (-2.5 / (x0*np.log(10))) * sig_x0x1

def calculate_sig_mu_th(z=z_cat, c=c_cat, sig_z=z_cat_err, sig_pec=0):    ## error from redshift measurements
    return np.sqrt((5 / (z * np.log(10)))**2 * ((sig_pec / c)**2 + sig_z**2))    # = (5 / np.log(10)) * (z_cat_err / z_cat)


# %%


## errors on hubble residuals: assume no errors on alpha and beta, and no correlations bewteen alpha and beta 
# x0, x1, c samples depend on which subpopulations we test (make sure [cond] used)
sig_x1 = x1_cat_err
sig_c = c_cat_err
sig_mB = calculate_sig_mB(x0=x0_cat, sig_x0=x0_cat_err)
sig_mBx1 = calculate_sig_mBx1(x0=x0_cat, sig_x0x1=sig_x0x1)
sig_mBc = calculate_sig_mBc(x0=x0_cat, sig_x0c=sig_x0c)

# this is independent of fitted nuisance parameters and can simply be added in quadrature to the error 
sig_mu_th = calculate_sig_mu_th(z=z_cat, c=c_cat, sig_z=z_cat_err, sig_pec=0)
var_th = torch.tensor(sig_mu_th**2)


# %%


def calculate_p_Im_vals(mass_chains, mass_mins, mass_maxs, M_thresh_val=10, n_walkers=256, n_gals=633, hi_mem=False, fixed=fixed): 
    """
    Inputs: mass_chains[cond] = SORTED MCMC mass chains for selected galaxies (i.e., populations) whose mass we care about;
                                (N_gals, 511488)
            mass_mins[cond] = selected minimum masses based on condition; (N_gals)
            mass_maxes[cond] = selected maximum masses based on condition; (N_gals) 
            M_thresh_val = torch.float OR torch.tensor 
                - if fixed: 10
                - if fitted: array of shape (N_walkers)
            n_walkers = number of MCMC walkers 
            n_gals = number of galaxies in subpopulation
    
    Outputs: (p_Im_0s, p_Im_1s) with shape (N_walkers, N_gal) and (narrow, fixed) ordering
    """
    # note mass_maxs, mass_mins dependent on selected galaxies 
    # for all others: have to do the computing step thing 
    p_Im_1s = torch.ones((n_walkers, n_gals), device=device) * -1 
    
    if fixed: 
        p_Im_1s[:, (mass_mins > M_thresh_val)] = 1           
        p_Im_1s[:, (mass_maxs < M_thresh_val)] = 0
        
        others_cond = ~((mass_mins > M_thresh_val) | (mass_maxs < M_thresh_val))
        p_Im_1s[:, others_cond] = torch.sum(mass_chains[others_cond] > M_thresh_val, dim=1) / mass_chains.shape[1]
    
    if not fixed:   # M_thresh is 
        assert M_thresh_val.ndim > 0 
        all_mass_mins = mass_mins.broadcast_to(n_walkers, -1)
        all_mass_maxs = mass_maxs.broadcast_to(n_walkers, -1)
        all_mass_chains = mass_chains.broadcast_to(n_walkers, -1, -1)

        p_Im_1s[all_mass_mins > M_thresh_val[:, None]] = 1          # M_thresh_val: (N_Walkers)  -> (N_walkers, 1)  
        p_Im_1s[all_mass_maxs < M_thresh_val[:, None]] = 0

        others_cond = ~((all_mass_mins > M_thresh_val[:, None]) | (all_mass_maxs < M_thresh_val[:, None]))
        
        # high-memory version 
        #### NOTE! COMPUTE THIS IN CHUNKS, IT WILL BE MUCH FASTER ^ ##### 
        if hi_mem: 
            M_thresh = M_thresh_val[:, None, None]
            p_Im_1s[others_cond] = torch.sum(all_mass_chains[others_cond, :] > M_thresh, dim=-1) / all_mass_chains[others_cond, :].shape[-1]
       
        # looped version
        else: 
            for i in tqdm(range(n_walkers)):                # for i in range(theta.shape[0]):   # 256
                # for each walker: fill in the columns that are other_cond
                ind_p_Im_1 = (torch.sum((mass_chains[(others_cond)[i]] > M_thresh_val[i]), dim=-1) / mass_chains.shape[1])
                p_Im_1s[i][others_cond[i]] = ind_p_Im_1

    p_Im_0s = 1 - p_Im_1s

    return p_Im_0s, p_Im_1s 


# %%


# log likelihood
def log_likelihood(theta, sn, sn_err, mu_model, mass_info, fixed=fixed):   
    """Inputs: 
    - theta = alpha, beta, gamma, M_B, sigma_int, (M_thresh); (N_walkers, N_params)
    - sn = mB, x1, c; (N_walkers, N_gals, 3), ordered within by (narrow, fixed) 
    - sn_err = sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th; (N_walkers, N_gals, 7) 
    - mu_model = predicted distance modulus from cosmology; (N_walkers, N_gals)
    - mass_info: dict; mass_chains, mass_mins, mass_maxs 
        - i) mass_chains = selected MCMC mass chains for given subpopulation; (N_walkers, N_gals). 
            - this WILL NOT be broadcasted in the fixed M_thresh case. 
            - this WILL be broadcasted in the fitted M_thresh case. 
        - ii) mass_mins = selected mass mins; (N_gals)
        - iii) mass_maxs = selected mass maxs; (N_gals)
    """
    n_walkers = theta.shape[0]
    n_gals = sn.shape[1]
    print('n_walkers:', n_walkers)
    print('n_gals:', n_gals)
    
    alpha = theta[:, 0][:, None]
    beta = theta[:, 1][:, None]
    gamma = theta[:, 2][:, None]      
    M_B = theta[:, 3][:, None] 
    sigma_int = theta[:, 4][:, None]
    print('alpha:', alpha.shape)          # (N_walkers, 1)   (one value for all galaxies fit)
    
    M_thresh_val = 10 if fixed else theta[:, 5]       # (N_walkers) 
    print('M_thresh_val:', M_thresh_val)

    # dict here 
    mass_chains = mass_info['mass_chains']
    mass_mins = mass_info['mass_mins']
    mass_maxs = mass_info['mass_maxs']
    print('mass_chains:', mass_chains.shape)
    print('mass_mins:', mass_mins.shape)
    
    p_Im_0, p_Im_1 = calculate_p_Im_vals(mass_chains, mass_mins, mass_maxs, M_thresh_val=M_thresh_val, 
                                             n_walkers=n_walkers, n_gals=n_gals, hi_mem=False, fixed=fixed)

    print('p_Im_0:', p_Im_0.shape)      # (N_walkers, N_gal) 
    print(p_Im_0)

    mB = sn[..., 0]                     # (N_walkers, N_gal) -> same, broadcasted for all walkers 
    x1 = sn[..., 1]
    c = sn[..., 2]
    print('mB:', mB.shape)
    
    mu_obs_lowmass = tripp(mB, x1, c, M_B, alpha, beta, gamma/2)          # (N_walkers, N_gal)
    mu_obs_highmass = tripp(mB, x1, c, M_B, alpha, beta, -gamma/2)
#     mu_model = mu_cat.broadcast_to(n_walkers, -1)                       # cosmology dependent; mu_cat is global 
    # 
    print('mu_obs_lowmass:', mu_obs_lowmass.shape)

    # dict here: sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th are global (depend on [cond])
    sig_mB = sn_err[..., 0]
    sig_x1 = sn_err[..., 1]
    sig_c = sn_err[..., 2]
    sig_mBx1 = sn_err[..., 3]
    sig_mBc = sn_err[..., 4]
    sig_x1c = sn_err[..., 5]
    var_th = sn_err[..., 6]
    print('sig_mB:', sig_mB.shape)
    
    var_obs = (sig_mB)**2 + alpha**2*(sig_x1)**2 + beta**2*(sig_c)**2 + 2*alpha*sig_mBx1 - 2*beta*sig_mBc - 2*alpha*beta*sig_x1c     # (N_walkers, N_gal)
#     var_obs_nocov = var_obs - 2*alpha*sig_mBx1 + 2*beta*sig_mBc + 2*alpha*beta*sig_x1c
#     var_obs = var_obs_nocov
    
    ## test with linalg to see if all covariances are positive definite
        # maybe only need to make sure determinant > 0 
        # if not, then issue with actual covariances themselves 
        # try with the original covariances as well and see if it's ztf that's the problem, or the mB conversion that is the problem 
    print('var_obs:', var_obs)
    print('# neg:', torch.unique(torch.where(var_obs < 0)[1]))
    print('neg:', var_obs[torch.where(var_obs < 0)])
    var_obs = torch.abs(var_obs)           # make sure positive 
    var_int = sigma_int**2
    var_mu = var_th + var_obs + var_int    # updated for each step; (N_walkers, N_gal)
    print('var_mu:', var_mu.shape)
    print('mu_model:', mu_model.shape)
    
    # fitting sig_int needs to include the normalization term for gaussian 
    norm = 1 / torch.sqrt(2 * torch.pi * var_mu)                                     # (N_walkers, N_gal)
    l0 = p_Im_0 * norm * torch.exp(-0.5 * (mu_obs_lowmass - mu_model)**2 / var_mu)    
    l1 = p_Im_1 * norm * torch.exp(-0.5 * (mu_obs_highmass - mu_model)**2 / var_mu)
    print('l0:', l0.shape)
    print('l1:', l1.shape)

    ll = torch.sum(torch.log(l0 + l1), dim=1)    
    print('ll:', ll.shape) 

    return ll   # shape (N_walkers)       


# %%


## priors
def log_prior(theta, fixed=fixed):  
    alpha = theta[:, 0]       
    beta = theta[:, 1]
    gamma = theta[:, 2]      
    M_B = theta[:, 3]
    sigma_int = theta[:, 4]
    
    if not fixed: 
        M_thresh = theta[:, 5]

    lp = torch.zeros(theta.shape[0], device=device)   # (N_walkers)

    if fixed: 
        bad = (
            (alpha < 0) | (alpha > 0.5) |
            (beta < 2) | (beta > 4.5) |
            (gamma < -0.5) | (gamma > 0.5) |
            (M_B < -21) | (M_B > -17) | 
            (sigma_int < 0.0) | (sigma_int > 1)
        )
    if not fixed: 
        bad = (
            (alpha < 0) | (alpha > 0.5) |
            (beta < 2) | (beta > 4.5) |
            (gamma < -0.5) | (gamma > 0.5) |
            (M_B < -21) | (M_B > -17) | 
            (sigma_int < 0.0) | (sigma_int > 1) | 
            (M_thresh > 9) | (M_thresh < 11)
        )

    lp[bad] = -torch.inf
    
    return lp      # (N_walkers)


# %%


## final log probability 
def log_prob(theta, sn, sn_err, mu_model, mass_info, fixed=fixed):
    print('theta:', theta.shape)
    print(theta)
    lp = log_prior(theta, fixed=fixed)
#     if not torch.isfinite(lp):
#         return -torch.inf
    print('lp:', lp.shape)
    ll = log_likelihood(theta, sn, sn_err, mu_model, mass_info, fixed=fixed)
    print('ll:', ll.shape)
    print('lp + ll:', lp + ll)
    return lp + ll


# %%


# initialization 
def init(sn, fixed=fixed): 
    # ztf dr2 volume-limited sample
    mB = sn[:, 0]
    x1 = sn[:, 1]
    c = sn[:, 2]

    alpha0 = 0.161
    beta0 = 3.05
    M_B0 = -19.36
    mu_obs0 = torch.tensor(tripp(mB, x1, c, M_B=M_B0, alpha=alpha0, beta=beta0, gamma=0))   # scatter doesn't change w/ addition of mass step 
    sig_int0 = torch.std(mu_obs0)   # maybe this is wrong bc it's sig_int should be AFTER standardization? 

    dir = 'ztf_meds/'
    HR = torch.tensor(np.load(os.path.join(base, dir, 'HR.npy')))            # all Hubble residuals
    HR_err = torch.tensor(np.load(os.path.join(base, dir, 'HR_err.npy')))     
    HR_invvar_weightmean = torch.sum(HR / HR_err**2) / torch.sum(1 / HR_err**2)
    gamma0 = 2 * HR_invvar_weightmean

    if not fixed: 
        M_thresh0 = 10
        return alpha0, beta0, gamma0, M_B0, sig_int0, M_thresh0
    else: 
        return alpha0, beta0, gamma0, M_B0, sig_int0


# %%
def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2,
           progress=True, save_lp=False, device=None):
    """
    Run affine invariant MCMC to sample a single posterior.
    """

    # Resolve device
    if device is None:
        if torch.is_tensor(walkers1):
            device = walkers1.device
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Progress-bar
    if progress:
        pbar = tqdm(total=n_steps, desc="Sampling")

    # Initialize current state
    current_state1 = torch.as_tensor(walkers1, device=device)
    current_state2 = torch.as_tensor(walkers2, device=device)

    # Initial target log prob for the walkers
    logp_current1 = torch.as_tensor(log_prob(current_state1), device=device)
    logp_current2 = torch.as_tensor(log_prob(current_state2), device=device)

    # sort out any nans
    logp_current1 = torch.where(
        torch.isnan(logp_current1),
        torch.ones_like(logp_current1, device=device).fill_(float("inf")),
        logp_current1,
    )
    logp_current2 = torch.where(
        torch.isnan(logp_current2),
        torch.ones_like(logp_current2, device=device).fill_(float("inf")),
        logp_current2,
    )

    # Holder for the whole chain
    chain = [torch.cat([current_state1, current_state2], axis=0)]
    if save_lp is True:
        lp_chain = [torch.cat([logp_current1, logp_current2], axis=0)]

    # MCMC loop
    for epoch in range(1, n_steps):

        # FIRST SET OF WALKERS:

        # Proposals
        idx1 = torch.randint(low=0, high=n_walkers, size=(n_walkers,), device=device)
        partners1 = current_state2[idx1]
        z1 = 0.5 * (torch.rand((n_walkers,), device=device) + 1) ** 2
        proposed_state1 = partners1 + (z1 * (current_state1 - partners1).T).T

        # Target log prob at proposed points
        logp_proposed1 = torch.as_tensor(log_prob(proposed_state1), device=device)
        logp_proposed1 = torch.where(
            torch.isnan(logp_proposed1),
            torch.ones_like(logp_proposed1, device=device).fill_(float("inf")),
            logp_proposed1,
        )

        # Acceptance probability
        p_accept1 = torch.minimum(
            torch.ones(n_walkers, device=device),
            z1 ** (n_params - 1) * torch.exp(logp_proposed1 - logp_current1),
        )

        # Accept or not
        accept1_ = torch.rand((n_walkers,), device=device) <= p_accept1
        accept1 = accept1_.to(dtype=current_state1.dtype)

        # Update the state
        current_state1 = (
            (current_state1).T * (1 - accept1) + (proposed_state1).T * accept1
        ).T
        logp_current1 = torch.where(accept1_, logp_proposed1, logp_current1)

        # SECOND SET OF WALKERS:

        # Proposals
        idx2 = torch.randint(low=0, high=n_walkers, size=(n_walkers,), device=device)
        partners2 = current_state1[idx2]
        z2 = 0.5 * (torch.rand((n_walkers,), device=device) + 1) ** 2
        proposed_state2 = partners2 + (z2 * (current_state2 - partners2).T).T

        # Target log prob at proposed points
        logp_proposed2 = torch.as_tensor(log_prob(proposed_state2), device=device)
        logp_proposed2 = torch.where(
            torch.isnan(logp_proposed2),
            torch.ones_like(logp_proposed2, device=device).fill_(float("inf")),
            logp_proposed2,
        )

        # Acceptance probability
        p_accept2 = torch.minimum(
            torch.ones(n_walkers, device=device),
            z2 ** (n_params - 1) * torch.exp(logp_proposed2 - logp_current2),
        )

        # Accept or not
        accept2_ = torch.rand((n_walkers,), device=device) <= p_accept2
        accept2 = accept2_.to(dtype=current_state2.dtype)

        # Update the state
        current_state2 = (
            (current_state2).T * (1 - accept2) + (proposed_state2).T * accept2
        ).T
        logp_current2 = torch.where(accept2_, logp_proposed2, logp_current2)

        # Append to chain
        chain.append(torch.cat([current_state1, current_state2], axis=0))
        if save_lp is True:
            lp_chain.append(torch.cat([logp_current1, logp_current2], axis=0))

        # Update the progressbar
        if progress:
            pbar.update(1)

    # Stack up the chain
    chain = torch.stack(chain, axis=0)
    if save_lp is True:
        lp_chain = torch.stack(lp_chain, axis=0)

    if save_lp is True:
        return chain, lp_chain
    else:
        return chain

# run mcmc 
def run_mcmc(sn, sn_err, mu_model, mass_info, n_params=5, n_walkers=256, n_steps=500, scale=0.1, fixed=True): 
    """Inputs: 
    - sn = mB, x1, c; (N_gals, 3), ordered within by (narrow, fixed) 
    - sn_err = sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th; (N_gals, 7)
    - mu_model = predicted distance modulus from cosmology; (N_gals)
    - mass_info = dict; mass_chains, mass_mins, mass_maxs 
        - i) mass_chains = selected MCMC mass chains for given subpopulation; (N_walkers, N_gals). 
            - this WILL NOT be broadcasted in the fixed M_thresh case. 
            - this WILL be broadcasted in the fitted M_thresh case. 
        - ii) mass_mins = selected mass mins; (N_gals)
        - iii) mass_maxs = selected mass maxs; (N_gals)
    - n_params = # params of fitted model: alpha, beta, gamma, M_B, sig_int, (M_thresh))
    """
    print('fixed:', fixed)
    p0 = torch.tensor(init(sn, fixed=fixed), device=device)    # gamma, a, sigma_int

    # two ensembles of walkers, which we'll start in a tight ball around the MAP estimate
    walkers1 = torch.randn(n_walkers, n_params, device=device)*scale + p0
    walkers2 = torch.randn(n_walkers, n_params, device=device)*scale + p0

    # input broadcasting here for all the walkers 
    sn_broadcast = sn.broadcast_to(n_walkers, -1, -1)
    sn_err_broadcast = sn_err.broadcast_to(n_walkers, -1, -1)
    mu_model_broadcast = mu_model.broadcast_to(n_walkers, -1)

    log_prob_func = lambda theta: log_prob(theta, sn_broadcast, sn_err_broadcast, mu_model_broadcast, mass_info,
                                           fixed=fixed)
    
    with torch.no_grad():
        mcmc_theta = sample(log_prob_func, n_params, n_walkers, n_steps, walkers1, walkers2, progress=True, save_lp=False, device=device)

    return mcmc_theta


# %%
### conditions: 
# cond_mass = (all)    
cond_ssfr = log10sSFR_med < -11
cond_dust = mcmc_dust2_med < 0.3
conds = {
        'cond_low_ssfr': cond_ssfr,
        'cond_high_ssfr': ~cond_ssfr,
        'cond_low_dust': cond_dust,
        'cond_high_dust': ~cond_dust
        }

cond_name = 'cond_high_dust'
cond = conds[cond_name] 
print(cond_name)

## run for 500 steps here 
sn = torch.stack((
    torch.tensor(mB_cat[cond], device=device),
    torch.tensor(x1_cat[cond], device=device),
    torch.tensor(c_cat[cond], device=device),
)).T

sn_err = torch.stack((
    torch.tensor(sig_mB[cond], device=device),
    torch.tensor(sig_x1[cond], device=device),
    torch.tensor(sig_c[cond], device=device),
    torch.tensor(sig_mBx1[cond], device=device),
    torch.tensor(sig_mBc[cond], device=device),
    torch.tensor(sig_x1c[cond], device=device),
    torch.tensor(var_th[cond], device=device),
)).T

mu_model = torch.tensor(mu_cat[cond], dtype=torch.float32, device=device)

mass_chains = mass_chains.to(device)
mass_mins = mass_mins.to(device)
mass_maxs = mass_maxs.to(device)
mass_info = {
    'mass_chains': mass_chains[cond],
    'mass_mins': mass_mins[cond],
    'mass_maxs': mass_maxs[cond]
}

n_params = 6
mcmc_theta = run_mcmc(sn, sn_err, mu_model, mass_info, 
                      n_params=n_params, n_walkers=256, n_steps=500, scale=0.1, fixed=False)

# n_params = 5
# mcmc_theta = run_mcmc(sn, sn_err, mu_model, mass_info, 
#                       n_params=n_params, n_walkers=256, n_steps=500, scale=0.1, fixed=True)


# %%


mcmc_theta_half2 = mcmc_theta.cpu().detach().numpy()[mcmc_theta.shape[0] // 2:].reshape(-1, n_params)
dir = 'ztf_sn_mcmc_chains/'
np.save(os.path.join(base, dir, f'theta_movable_M_thresh_{cond_name}.npy'), mcmc_theta_half2)
print(np.median(mcmc_theta_half2, axis=0))


# %%


# alpha_fit = mcmc_theta_half2[:, 0]
# beta_fit = mcmc_theta_half2[:, 1]
# gamma_fit = mcmc_theta_half2[:, 2]
# M_B_fit = mcmc_theta_half2[:, 3]
# sigma_int_fit = mcmc_theta_half2[:, 4]

# np.median(mcmc_theta_half2, axis=0)

# # names = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$M_B$', r'$\sigma_{int}$']  
# names = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$M_B$', r'$\sigma_{int}$', r'$M_{thresh}'] 

# figure = corner.corner(mcmc_theta_half2,    # only the second half of samples
#                        labels=names, smooth=0.55, 
#                        smooth1d=0.5, plot_density=True, 
#                        fill_contours=True, plot_datapoints=False,
#                        color='#EE6677', 
#                                label_kwargs={'fontsize':18})

# alpha0 = 0.161
# beta0 = 3.05
# M_B0 = -19.36
# mu_obs0 = torch.tensor(tripp(mB_cat, x1_cat, c_cat, M_B=M_B0, alpha=alpha0, beta=beta0, gamma=0)) 


# plt.plot(np.sort(z_cat), mu_obs0[np.argsort(z_cat)], 'o', label='ztf-predicted')
# plt.plot(np.sort(z_cat), mu_cat[np.argsort(z_cat)], '.', label='model')

# # masking with gamma based on mass: 
# mu_pred0 = tripp(mB_cat[np.argsort(z_cat)], x1_cat[np.argsort(z_cat)], c_cat[np.argsort(z_cat)], 
#                       M_B=np.median(M_B_fit), alpha=np.median(alpha_fit), beta=np.median(beta_fit), gamma=0)
# plt.plot(np.sort(z_cat), mu_pred0, '.', 
#          label=r'fit-values, no $\gamma$')
# plt.yscale('log')
# plt.legend()

# np.std(mu_pred0)

# # test the mass step 
# cond_mass = log10M_remain_med < 10 
# mu_pred = tripp(mB_cat[np.argsort(z_cat)], x1_cat[np.argsort(z_cat)], c_cat[np.argsort(z_cat)], 
#                       M_B=np.median(M_B_fit), alpha=np.median(alpha_fit), beta=np.median(beta_fit), gamma=0)

# mu_pred[cond_mass] = mu_pred[cond_mass] + np.median(gamma_fit) / 2
# mu_pred[~cond_mass] = mu_pred[~cond_mass] - np.median(gamma_fit) / 2


# plt.plot(np.sort(z_cat), mu_obs0[np.argsort(z_cat)], 'o')
# plt.plot(np.sort(z_cat), mu_cat[np.argsort(z_cat)], '.')

# # masking with gamma based on mass: 
# plt.plot(np.sort(z_cat), mu_pred, '.')
# plt.yscale('log')

# np.std(mu_pred) # weird bc this should in theory reduce the scatter?

# ## plot mass-step corrected and mass-step uncorrected from fits
# plt.plot(mu_pred0, mu_pred, 'o')
# plt.xlabel(r'fit-values, $\gamma$ uncorrected')
# plt.ylabel(r'fit-values, $\gamma$ corrected')

# hr_fit = mu_pred - mu_cat

# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(log10M_remain_med, hr_fit, 'o')
# ax.axvline(10, color='black', ls='--')
# xdiv = (10 - log10M_remain_med.min()) / (log10M_remain_med.max() - log10M_remain_med.min())
# xdiv = xdiv - 0.02
# ax.axhline(np.median(gamma_fit) / 2, xmin=0, xmax=xdiv, color='firebrick', ls='--')
# ax.axhline(- np.median(gamma_fit) / 2, xmin=xdiv, xmax=1, color='firebrick', ls='--')
# ax.set_title('mass step', fontsize=15)


# %%





# %%




