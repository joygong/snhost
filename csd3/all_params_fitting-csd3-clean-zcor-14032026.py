#!/usr/bin/env python
# coding: utf-8
# %%

# %%
### this version is made for csd3 ### 

# %%


import os
from turtle import tracer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sncosmo
import astropy
import astropy.constants as const
from astropy.cosmology.realizations import Planck18
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import sklearn
import scipy.stats
import itertools

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

# %%
base = '/home/jjg65/rds/hpc-work/'
dir = 'ztf_sn/'
# base = './'
# dir = ''

ztf_snia_vollim = pd.read_csv(base + dir + 'ztf_snia_vollim.csv')
ztf_snia_vollim_zcmb_zcor = pd.read_csv(base + dir + 'ztf_snia_vollim_zcmb_zcor.csv')
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

## here flow-corrected zcmb values 
zcmb_cat = np.concatenate((ztf_snia_vollim_zcmb_zcor['zcmb_cor'][cond_narrow].values, 
                           ztf_snia_vollim_zcmb_zcor['zcmb_cor'][cond_fixed].values))
zcmb_cat_err = np.concatenate((ztf_snia_vollim_zcmb_zcor['zcmb_err'][cond_narrow].values, 
                               ztf_snia_vollim_zcmb_zcor['zcmb_err'][cond_fixed].values))
                               # error is NOT from pec z, ONLY from original redshift error + helio conversion (error in v_sun)


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
zcmb_cat = torch.tensor(zcmb_cat)
mu_cat = cosmo.distmod(zcmb_cat).value



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

def calculate_sig_mu_th(z=zcmb_cat, sig_z=zcmb_cat_err, sig_pec=200):    ## error from redshift measurements
    """z = (flow/peculiar-velocity CORRECTED) rest frame cmb redshift; sig_z = rest frame cmb redshift error 
    sig_pec (fix at 200 km/s), c = speed of light (km/s) 
    """
    c = (const.c).to(u.km / u.s).value
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
# sig_mu_th = calculate_sig_mu_th(z=z_cat, sig_z=z_cat_err, sig_pec=0)
# var_th = torch.tensor(sig_mu_th**2)

### here are the flow-corrected cmb restframe redshifts, and using a fixed sig_pec 
sig_mu_th = calculate_sig_mu_th(z=zcmb_cat, sig_z=zcmb_cat_err, sig_pec=200)   
var_th = torch.tensor(sig_mu_th**2)


# %%


def calculate_p_Im_vals(tracer_chains, tracer_mins, tracer_maxs, tracer_thresh_val, fixed, n_walkers=256, n_gals=633, hi_mem=False): 
    """
    Inputs: tracer_chains[cond] = SORTED MCMC tracer [mass, sSFR, or dust2] chains for selected galaxies (i.e., populations) whose tracer we care about;
                                (N_gals, 511488)
            tracer_mins[cond] = selected minimum tracer based on condition; (N_gals)
            tracer_maxes[cond] = selected maximum tracer based on condition; (N_gals) 
            tracer_thresh_val = torch.float OR torch.tensor 
                - if fixed: 10 [mass]; -11 [log10sSFR]; 0.6 [dust2]
                - if fitted: array of shape (N_walkers)
            n_walkers = number of MCMC walkers 
            n_gals = number of galaxies in subpopulation
    
    Outputs: (p_Im_0s, p_Im_1s) with shape (N_walkers, N_gal) and (narrow, fixed) ordering
    """
    # note mass_maxs, mass_mins dependent on selected galaxies 
    # for all others: have to do the computing step thing 
    assert tracer_chains.shape == torch.Size([633, 511488])
    
    p_Im_1s = torch.ones((n_walkers, n_gals), device=device) * -1 
    
    if fixed: 
        p_Im_1s[:, (tracer_mins > tracer_thresh_val)] = 1           
        p_Im_1s[:, (tracer_maxs < tracer_thresh_val)] = 0
        
        others_cond = ~((tracer_mins > tracer_thresh_val) | (tracer_maxs < tracer_thresh_val))
        p_Im_1s[:, others_cond] = torch.sum(tracer_chains[others_cond] > tracer_thresh_val, dim=1) / tracer_chains.shape[1]
    
    if not fixed:   # M_thresh is 
        assert tracer_thresh_val.ndim > 0 
        all_tracer_mins = tracer_mins.broadcast_to(n_walkers, -1)
        all_tracer_maxs = tracer_maxs.broadcast_to(n_walkers, -1)
        all_tracer_chains = tracer_chains.broadcast_to(n_walkers, -1, -1)

        p_Im_1s[all_tracer_mins > tracer_thresh_val[:, None]] = 1          # M_thresh_val: (N_Walkers)  -> (N_walkers, 1)  
        p_Im_1s[all_tracer_maxs < tracer_thresh_val[:, None]] = 0

        others_cond = ~((all_tracer_mins > tracer_thresh_val[:, None]) | (all_tracer_maxs < tracer_thresh_val[:, None]))
        
        # high-memory version 
        #### NOTE! COMPUTE THIS IN CHUNKS, IT WILL BE MUCH FASTER ^ ##### 
        if hi_mem: 
            tracer_thresh = tracer_thresh_val[:, None, None]
            p_Im_1s[others_cond] = torch.sum(all_tracer_chains[others_cond, :] > tracer_thresh, dim=-1) / all_tracer_chains[others_cond, :].shape[-1]
       
        # looped version
        else: 
            for i in tqdm(range(n_walkers)):                # for i in range(theta.shape[0]):   # 256
                # for each walker: fill in the columns that are other_cond
                ind_p_Im_1 = (torch.sum((tracer_chains[(others_cond)[i]] > tracer_thresh_val[i]), dim=-1) / tracer_chains.shape[1])
                p_Im_1s[i][others_cond[i]] = ind_p_Im_1

    p_Im_0s = 1 - p_Im_1s

    return p_Im_0s, p_Im_1s 


# %%
### start new index function 
def get_param_indices(tracer_name, fixed, alpha_split, beta_split):
    i = 0
    idx = {}
    idx['alpha_low'] = i; i += 2 if alpha_split else 1
    idx['alpha_high'] = i - 1 if alpha_split else None
    idx['beta_low'] = i; i += 2 if beta_split else 1
    idx['beta_high'] = i - 1 if beta_split else None
    idx['gamma'] = i; i += 1
    idx['M_B'] = i; i += 1
    idx['sigma_int'] = i; i += 1
    idx[f'{tracer_name}_thresh'] = i if not fixed else None
    return idx
### end new index function 

# %%


# log likelihood
def log_likelihood(theta, sn, sn_err, mu_model, tracer_info, fixed, alpha_split, beta_split, covar):   
    """Inputs: 
    - theta = alpha, beta, gamma, M_B, sigma_int, (M_thresh); (N_walkers, N_params)
    - sn = mB, x1, c; (N_walkers, N_gals, 3), ordered within by (narrow, fixed) 
    - sn_err = sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th; (N_walkers, N_gals, 7) 
    - mu_model = predicted distance modulus from cosmology; (N_walkers, N_gals)
    - tracer_info: dict; tracer_chains, tracer_mins, tracer_maxs; [mass, sSFR, or dust2]
        - i) tracer_chains = selected MCMC mass chains for given subpopulation; (N_walkers, N_gals). 
            - this WILL NOT be broadcasted in the fixed tracer_thresh case. 
            - this WILL be broadcasted in the fitted M_thresh case. 
        - ii) tracer_mins = selected mass mins; (N_gals)
        - iii) tracer_maxs = selected mass maxs; (N_gals)
    """
    n_walkers = theta.shape[0]
    n_gals = sn.shape[1]
    tracer_name = tracer_info['tracer_name']   # should be global

    print('n_walkers:', n_walkers)
    print('n_gals:', n_gals)
    print('tracer_name:', tracer_name)
    
    # alpha (use alpha_low; if not split, alpha_high is None and you use the same value)
    idx = get_param_indices(tracer_name, fixed, alpha_split, beta_split)
    alpha_low = theta[:, idx['alpha_low']][:, None]
    print('alpha_low:', idx['alpha_low'])
    alpha_high = theta[:, idx['alpha_high']][:, None] if idx['alpha_high'] is not None else alpha_low
    print('alpha_high:', idx['alpha_high'])

    # beta
    beta_low = theta[:, idx['beta_low']][:, None]
    print('beta_low:', idx['beta_low'])
    beta_high = theta[:, idx['beta_high']][:, None] if idx['beta_high'] is not None else beta_low
    print('beta_high:', idx['beta_high'])
    
    # gamma, M_B, sigma_int
    gamma = theta[:, idx['gamma']][:, None]
    print('gamma:', idx['gamma'])
    M_B = theta[:, idx['M_B']][:, None]
    print('M_B:', idx['M_B'])
    sigma_int = theta[:, idx['sigma_int']][:, None]
    print('sigma_int:', idx['sigma_int'])

    # tracer_thresh (only when not fixed)
    tracer_thresh_vals = {
        'mass': 10,
        'sSFR': -11,
        'dust2': 0.6         # 0.3*(3.05 - 1)   
    }
    tracer_thresh_val = theta[:, idx[f'{tracer_name}_thresh']] if idx[f'{tracer_name}_thresh'] is not None else tracer_thresh_vals[tracer_name]
    print('tracer_thresh_val:', idx[f'{tracer_name}_thresh'])
    print(f'{tracer_name}_thresh_val:', tracer_thresh_val)
    print('alpha_low:', alpha_low.shape)          # (N_walkers, 1)   (one value for all galaxies fit)

    # dict here 
    tracer_chains = tracer_info['tracer_chains']
    tracer_mins = tracer_info['tracer_mins']
    tracer_maxs = tracer_info['tracer_maxs']
    print('tracer_chains:', tracer_chains.shape)
    print('tracer_mins:', tracer_mins.shape)
    
    p_Im_0, p_Im_1 = calculate_p_Im_vals(tracer_chains, tracer_mins, tracer_maxs, tracer_thresh_val,
                                             fixed, n_walkers=n_walkers, n_gals=n_gals, hi_mem=False)

    print('p_Im_0:', p_Im_0.shape)      # (N_walkers, N_gal) 
    print(p_Im_0)

    mB = sn[..., 0]                     # (N_walkers, N_gal) -> same, broadcasted for all walkers 
    x1 = sn[..., 1]
    c = sn[..., 2]
    print('mB:', mB.shape)
    
    ######### distinct populations ###########
    # alpha, beta per tracer
#     alpha_lo, alpha_hi = {                   # note: (lo, hi) are "fake" defs & can be (1, 2); (low, high) are "true" param low/high defs in VALUE
#         "mass":  (alpha_low,  alpha_high),
#         "sSFR":  (alpha_high, alpha_low),
#         "dust2": (alpha_low,  alpha_high),
#     }[tracer_name]

#     beta_lo, beta_hi = {
#         "mass":  (beta_low,  beta_high),
#         "sSFR":  (beta_high, beta_low),
#         "dust2": (beta_low,  beta_high),
#     }[tracer_name]

    # gamma per tracer
        # note these are in the opposite way of the correction, so fitted vals should be negative
#     gamma_lo, gamma_hi = {
#         "mass":  ( gamma/2, -gamma/2),         # assume after alpha, beta stand, high mass are brighter (have neg HR) 
#         "sSFR":  (-gamma/2,  gamma/2),         # assume after alpha, beta stand, low sSFR are brighter (have neg HR)
#         "dust2": ( gamma/2, -gamma/2),         # assume after alpha, beta stand, high dusty are brighter (have neg HR)
#     }[tracer_name]
    
    # (THIS IS CORRECT): low mass, star-forming hosts have a brighter intrinsic absolute magnitude (M0 - gamma/2 vs. M0 + gamma/2)
    gamma_lo, gamma_hi = {
        "mass":  (-gamma/2, gamma/2),            # low mass hosts intrinsically host brighter sne
        "sSFR":  (gamma/2,  -gamma/2),           # high ssfr hosts intrinsically host brighter sne
        "dust2": (-gamma/2, gamma/2),            # dust-poor hosts intrinsically host brighter sne 
    }[tracer_name]
    
    # evaluate distance modulus
#     mu_obs_lo = tripp(mB, x1, c, M_B, alpha_lo, beta_lo, gamma_lo)      # (N_walkers, N_gal)
#     mu_obs_hi = tripp(mB, x1, c, M_B, alpha_hi, beta_hi, gamma_hi)
    
    # to make interpretation easier, we can write alpha_low, alpha_high, beta_low, beta_high for all. this corresponds to low//high in TRACER VALUE rather than in ALPHA/BETA VALUE.
    mu_obs_lo = tripp(mB, x1, c, M_B, alpha_low, beta_low, gamma_lo)      # (N_walkers, N_gal)
    mu_obs_hi = tripp(mB, x1, c, M_B, alpha_high, beta_high, gamma_hi)


#     mu_model = mu_cat.broadcast_to(n_walkers, -1)                       # cosmology dependent; mu_cat is global 
    print(f'mu_obs_lo{tracer_name}:', mu_obs_lo.shape)

    # dict here: sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th are global (depend on [cond])
    sig_mB = sn_err[..., 0]
    sig_x1 = sn_err[..., 1]
    sig_c = sn_err[..., 2]
    sig_mBx1 = sn_err[..., 3]
    sig_mBc = sn_err[..., 4]
    sig_x1c = sn_err[..., 5]
    var_th = sn_err[..., 6]
    print('sig_mB:', sig_mB.shape)
    
    # var_obs needs to be per-population when alpha or beta is split:
    var_obs_lo = (sig_mB)**2 + alpha_low**2*(sig_x1)**2 + beta_low**2*(sig_c)**2 + 2*alpha_low*sig_mBx1 - 2*beta_low*sig_mBc - 2*alpha_low*beta_low*sig_x1c
    var_obs_hi = (sig_mB)**2 + alpha_high**2*(sig_x1)**2 + beta_high**2*(sig_c)**2 + 2*alpha_high*sig_mBx1 - 2*beta_high*sig_mBc - 2*alpha_high*beta_high*sig_x1c    # (N_walkers, N_gal)
    if not covar: 
        var_obs_lo = (sig_mB)**2 + alpha_low**2*(sig_x1)**2 + beta_low**2*(sig_c)**2 
        var_obs_hi = (sig_mB)**2 + alpha_high**2*(sig_x1)**2 + beta_high**2*(sig_c)**2 
        
#     var_obs_lo = (sig_mB)**2 + alpha_lo**2*(sig_x1)**2 + beta_lo**2*(sig_c)**2 + 2*alpha_lo*sig_mBx1 - 2*beta_lo*sig_mBc - 2*alpha_lo*beta_lo*sig_x1c
#     var_obs_hi = (sig_mB)**2 + alpha_hi**2*(sig_x1)**2 + beta_hi**2*(sig_c)**2 + 2*alpha_hi*sig_mBx1 - 2*beta_hi*sig_mBc - 2*alpha_hi*beta_hi*sig_x1c    # (N_walkers, N_gal)
#     if not covar: 
#         var_obs_lo = (sig_mB)**2 + alpha_lo**2*(sig_x1)**2 + beta_lo**2*(sig_c)**2 
#         var_obs_hi = (sig_mB)**2 + alpha_hi**2*(sig_x1)**2 + beta_hi**2*(sig_c)**2 
    print('covar:', covar)
    print('var_obs_lo:', var_obs_lo)
    print('# neg lo:', torch.unique(torch.where(var_obs_lo < 0)[1]))
    print('neg:', var_obs_lo[torch.where(var_obs_lo < 0)])
    
    print('var_obs_hi:', var_obs_hi)
    print('# neg hi:', torch.unique(torch.where(var_obs_hi < 0)[1]))
    print('neg:', var_obs_hi[torch.where(var_obs_hi < 0)])

    ## test with linalg to see if all covariances are positive definite
        # maybe only need to make sure determinant > 0 
        # if not, then issue with actual covariances themselves 
        # try with the original covariances as well and see if it's ztf that's the problem, or the mB conversion that is the problem 
    
    var_obs_lo = torch.abs(var_obs_lo)           # make sure positive 
    var_obs_hi = torch.abs(var_obs_hi)
    var_int = sigma_int**2
    var_mu_lo = var_th + var_obs_lo + var_int
    var_mu_hi = var_th + var_obs_hi + var_int
    print('var_mu_lo:', var_mu_lo.shape)
    print('mu_model:', mu_model.shape)
    
    # fitting sig_int needs to include the normalization term for gaussian 
    norm_lo = 1 / torch.sqrt(2 * torch.pi * var_mu_lo)
    norm_hi = 1 / torch.sqrt(2 * torch.pi * var_mu_hi)
    if (not alpha_split) and (not beta_split):
        assert torch.equal(norm_lo, norm_hi)
        assert torch.equal(var_mu_lo, var_mu_hi)
    l0 = p_Im_0 * norm_lo * torch.exp(-0.5 * (mu_obs_lo - mu_model)**2 / var_mu_lo)
    l1 = p_Im_1 * norm_hi * torch.exp(-0.5 * (mu_obs_hi - mu_model)**2 / var_mu_hi)
    print('l0:', l0.shape)
    print('l1:', l1.shape)

    ll = torch.sum(torch.log(l0 + l1), dim=1)    
    print('ll:', ll.shape) 

    return ll   # shape (N_walkers)       


# %%


## priors
def log_prior(theta, tracer_name, fixed, alpha_split, beta_split):       # tracer_name is global -> maybe change this to info to be consistent 
    
    idx = get_param_indices(tracer_name, fixed, alpha_split, beta_split)
    
    alpha_low = theta[:, idx['alpha_low']]
    alpha_high = theta[:, idx['alpha_high']] if idx['alpha_high'] is not None else alpha_low
    beta_low = theta[:, idx['beta_low']]
    beta_high = theta[:, idx['beta_high']] if idx['beta_high'] is not None else beta_low
    gamma = theta[:, idx['gamma']]
    M_B = theta[:, idx['M_B']]
    sigma_int = theta[:, idx['sigma_int']]
    if not fixed:   # if idx[f'{tracer_name}_thresh'] is not None 
        tracer_thresh = theta[:, idx[f'{tracer_name}_thresh']] 
    else: 
        tracer_thresh = None       
   
    bad_alpha = (
        (alpha_low  < 0) | (alpha_low  > 0.5) |
        (alpha_high < 0) | (alpha_high > 0.5)
    )
    
    bad_beta = (
        (beta_low  < 2) | (beta_low  > 4.5) |
        (beta_high < 2) | (beta_high > 4.5)
    )
    
    bad_shared = (
        (gamma < -0.5) | (gamma > 0.5) |
        (M_B < -21)    | (M_B > -17) |
        (sigma_int < 0.0) | (sigma_int > 1.0)
    )

    if not fixed: 
        tracer_bounds = {
            'mass':  (9, 11),
            'sSFR':  (-12, -9),
            'dust2': (0.5, 0.8)    # originally [0.3, 1.0] 
        }

        lo, hi = tracer_bounds[tracer_name]
        bad_tracer = (tracer_thresh < lo) | (tracer_thresh > hi)
        
    else:
        bad_tracer = torch.zeros(theta.shape[0], dtype=torch.bool, device=device)
        
    bad = bad_alpha | bad_beta | bad_shared | bad_tracer
    print('bad:', bad)

    lp = torch.zeros(theta.shape[0], device=device)   # (N_walkers)
    lp[bad] = -torch.inf
    
    return lp      # (N_walkers)


# %%


## final log probability 
def log_prob(theta, sn, sn_err, mu_model, tracer_info, fixed, alpha_split, beta_split, covar):   # tracer_name is global 
    print('theta:', theta.shape)
    print(theta)
    tracer_name = tracer_info['tracer_name']
    lp = log_prior(theta, tracer_name, fixed, alpha_split, beta_split)
#     if not torch.isfinite(lp):
#         return -torch.inf
    print('lp:', lp.shape)
    ll = log_likelihood(theta, sn, sn_err, mu_model, tracer_info, fixed, alpha_split, beta_split, covar)
    print('ll:', ll.shape)
    print('lp + ll:', lp + ll)
    return lp + ll


# %%


# initialization 
def init(sn, tracer_name, fixed, alpha_split, beta_split): 
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
#     dir = ''
    HR = torch.tensor(np.load(os.path.join(base, dir, 'HR.npy')))            # all Hubble residuals
    HR_err = torch.tensor(np.load(os.path.join(base, dir, 'HR_err.npy')))    
    HR_invvar_weightmean = torch.sum(HR / HR_err**2) / torch.sum(1 / HR_err**2)
    gamma0 = 2 * HR_invvar_weightmean
    
    params = []
    if alpha_split:
        params += [alpha0, alpha0]   # alpha_low, alpha_high
    else:
        params += [alpha0]           # single alpha

    if beta_split:
        params += [beta0, beta0]     # beta_low, beta_high
    else:
        params += [beta0]

    params += [gamma0, M_B0, sig_int0]
#     params += [0, M_B0, sig_int0]
    
    if not fixed:
        tracer_thresh0 = {
            'mass': 10.0,
            'sSFR': -11.0,
            'dust2': 0.6     # 0.3*(3.05 - 1)  
        }
        params += [tracer_thresh0[tracer_name]]

    return tuple(torch.tensor(params))


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


# %%


# run mcmc 
def run_mcmc(sn, sn_err, mu_model, tracer_info, n_params, n_walkers, n_steps, scale, fixed, alpha_split, beta_split, covar): 
    """Inputs: 
    - sn = mB, x1, c; (N_gals, 3), ordered within by (narrow, fixed) 
    - sn_err = sig_mB, sig_x1, sig_c, sig_mBx1, sig_mBc, sig_x1c, var_th; (N_gals, 7)
    - mu_model = predicted distance modulus from cosmology; (N_gals)
     - tracer_info: dict; tracer_chains, tracer_mins, tracer_maxs; [mass, sSFR, or dust2]
        - i) tracer_chains = selected MCMC mass chains for given subpopulation; (N_walkers, N_gals). 
            - this WILL NOT be broadcasted in the fixed tracer_thresh case. 
            - this WILL be broadcasted in the fitted M_thresh case. 
        - ii) tracer_mins = selected mass mins; (N_gals)
        - iii) tracer_maxs = selected mass maxs; (N_gals)
    - n_params = # params of fitted model: alpha, beta, gamma, M_B, sig_int, (M_thresh))
    """
    print('fixed:', fixed)
    p0 = torch.tensor(init(sn, tracer_info['tracer_name'], fixed, alpha_split, beta_split), device=device)    
    print('p0:', p0)
    
    # two ensembles of walkers, which we'll start in a tight ball around the MAP estimate
    walkers1 = torch.randn(n_walkers, n_params, device=device)*scale + p0
    walkers2 = torch.randn(n_walkers, n_params, device=device)*scale + p0

    # input broadcasting here for all the walkers 
    sn_broadcast = sn.broadcast_to(n_walkers, -1, -1)
    sn_err_broadcast = sn_err.broadcast_to(n_walkers, -1, -1)
    mu_model_broadcast = mu_model.broadcast_to(n_walkers, -1)

    log_prob_func = lambda theta: log_prob(theta, sn_broadcast, sn_err_broadcast, mu_model_broadcast, tracer_info,
                                           fixed, alpha_split, beta_split, covar)
    
    with torch.no_grad():
        mcmc_theta = sample(log_prob_func, n_params, n_walkers, n_steps, walkers1, walkers2, progress=True, save_lp=False, device=device)

    return mcmc_theta


# %%
### conditions: 
cond_nosplit = np.arange(633)
cond = cond_nosplit

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


# %%

long_names = {
    'mass': 'log10M_remain',
    'sSFR': 'log10sSFR',
     'dust2': 'mcmc_dust2'
}
dir = 'ztf_meds'
# dir = ''

# tracers = ["mass", "sSFR", "dust2"]
tracers = ["dust2"]

for tracer_name in tracers:
    dir = 'ztf_meds/'
#     dir = '' 
    
    tracer_chains = torch.load(base + dir + f'{long_names[tracer_name]}_chains_sorted.pt').to(device)
    tracer_mins = torch.load(base + dir + f'{tracer_name}_mins.pt').to(device)
    tracer_maxs = torch.load(base + dir + f'{tracer_name}_maxs.pt').to(device)
    
    tracer_info = {
    'tracer_name': tracer_name,
    'tracer_chains': tracer_chains[cond],
    'tracer_mins': tracer_mins[cond],
    'tracer_maxs': tracer_maxs[cond]
    }
    
    saved_fixed_pim = False
    
    for fixed, alpha_split, beta_split in itertools.product(
        [True, False],
        [False, True],
        [False, True]
    ):

        print("=" * 70)
        print(f"Running case:")
        print(f"  tracer_name = {tracer_name}")
        print(f"  fixed       = {fixed}")
        print(f"  alpha_split = {alpha_split}")
        print(f"  beta_split  = {beta_split}")
        print("=" * 70)
        
        idx = get_param_indices(tracer_name, fixed, alpha_split, beta_split)
        n_params = idx[f'{tracer_name}_thresh'] + 1 if idx[f'{tracer_name}_thresh'] is not None else idx['sigma_int'] + 1
        print('n_params:', n_params)
        n_walkers = 256
        n_steps = 500
        scale = 0.1
        mcmc_theta = run_mcmc(sn, sn_err, mu_model, tracer_info, 
                              n_params, n_walkers, n_steps, scale, fixed, 
                              alpha_split, beta_split, covar=True)
        print('mcmc_theta shape:', mcmc_theta.shape)

        mcmc_theta_half2 = mcmc_theta.cpu().detach().numpy()[mcmc_theta.shape[0] // 2:].reshape(-1, n_params)
        print('mcmc_theta_half2 shape:', mcmc_theta_half2.shape)
        print(np.median(mcmc_theta_half2, axis=0)), print(np.std(mcmc_theta_half2, axis=0))
        
        ### save everything 
        model_tag = [
            tracer_name,
            "fix" if fixed else "mov",
            "a" if alpha_split else None,
            "b" if beta_split else None,
        ]

        filename = "_".join([t for t in model_tag if t]) + "_chain.npy"

        dir = 'ztf_sn_mcmc_chains/'
#         dir = ''
        path = 'all_params_fitting/zcor_14032026/'
        np.save(os.path.join(base, dir, path, filename), mcmc_theta_half2)

        
        # FIXED THRESHOLD: save p_Im_1 only once (note: this is independent of fitted parameters) 
        if fixed and not saved_fixed_pim:

            tracer_thresh_vals = {"mass":10, "sSFR":-11, "dust2":0.6}
            tracer_thresh_val = tracer_thresh_vals[tracer_name]

            n_gals = tracer_info["tracer_chains"].shape[0]

            p_Im_0, p_Im_1 = calculate_p_Im_vals(
                tracer_info["tracer_chains"],
                tracer_info["tracer_mins"],
                tracer_info["tracer_maxs"],
                tracer_thresh_val,
                fixed=True,
                n_walkers=1,
                n_gals=n_gals,
                hi_mem=False
            )

            path = 'all_params_fitting/zcor_14032026/p_Im_1/'

            np.save(
                os.path.join(base, dir, path, f"{tracer_name}_fixed_pIm1.npy"),
                p_Im_1.cpu().numpy()
            )

            saved_fixed_pim = True


        # MOVABLE THRESHOLD: save every fit
        elif not fixed:

            theta_final = torch.tensor(mcmc_theta_half2, device=device)

            idx = get_param_indices(tracer_name, fixed, alpha_split, beta_split)
            tracer_thresh_val = theta_final[:, idx[f'{tracer_name}_thresh']]

            n_walkers = theta_final.shape[0]
            n_gals = tracer_info["tracer_chains"].shape[0]

            p_Im_0, p_Im_1 = calculate_p_Im_vals(
                tracer_info["tracer_chains"],
                tracer_info["tracer_mins"],
                tracer_info["tracer_maxs"],
                tracer_thresh_val,
                fixed=False,
                n_walkers=n_walkers,
                n_gals=n_gals,
                hi_mem=False
            )

            path = 'all_params_fitting/zcor_14032026/p_Im_1/'

            np.save(
                os.path.join(base, dir, path, filename.replace("_chain.npy","_pIm1.npy")),
                p_Im_1.cpu().numpy()
            )
 
        print("\n")
        


# %%
### note: if i split alpha by x1 or beta by c, 
    # i still have a weighting but it's not an "integral" approx anymore: 
    # it would just be a EITHER 0 OR 1 FOR EVERY SUPERNOVA (the x1 is "deterministic" rather than probabilistic)
    

### OR I COULD INCORPORATE THE ERROR ON X1 SOMEHOW: 
    # ztf_snia_vollim_zcmb_zcor['x1_err'], this would either result in: 
        # 1) 0.5 * pop_lo + 0.5 * pop_hi,
        # 2) 1 * pop_lo + 0 * pop_hi,
        # 3) 0 * pop_lo + 1 * pop_hi. 

# %%
### post processing of this 

# long_names = {
# 'mass': 'log10M_remain',
# 'sSFR': 'log10sSFR',
#  'dust2': 'mcmc_dust2'
# }

# dir = ''
# tracer_name = 'dust2'
# tracer_chains = torch.load(base + dir + f'{long_names[tracer_name]}_chains_sorted.pt').to(device)
# tracer_mins = torch.load(base + dir + f'{tracer_name}_mins.pt').to(device)
# tracer_maxs = torch.load(base + dir + f'{tracer_name}_maxs.pt').to(device)

# # for each gal have a weight -> at the end average all to see if weight for tracer thresh is ~0.5? 
# n_walkers = 256
# n_gals = 633
# p_Im_1s = torch.ones((n_walkers, n_gals), device=device) * -1 
# fixed = True

# # tracer_thresh_val = 0.70263652
# tracer_thresh_val = 0.5

# if fixed: 
#     p_Im_1s[:, (tracer_mins > tracer_thresh_val)] = 1           
#     p_Im_1s[:, (tracer_maxs < tracer_thresh_val)] = 0

#     others_cond = ~((tracer_mins > tracer_thresh_val) | (tracer_maxs < tracer_thresh_val))
#     p_Im_1s[:, others_cond] = torch.sum(tracer_chains[others_cond] > tracer_thresh_val, dim=1) / tracer_chains.shape[1]

# %%

# %%
