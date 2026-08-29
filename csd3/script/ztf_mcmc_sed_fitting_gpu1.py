#!/usr/bin/env python
# coding: utf-8
# %%
print('importing...')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.cosmology.realizations import Planck18

import torch
import pickle
import corner
from tqdm import tqdm
from affine import sample, sample_batch
import speculator
from speculator import PhotulatorModelStack

from pop_cosmos.catalogue import CatalogueGenerator
from pop_cosmos.utils import compute_derived_quantities, compute_mass_remaining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

# %%
torch.set_default_dtype(torch.float32)
# torch.serialization.add_safe_globals([speculator.speculator.Photulator])
# torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterList])

# %%
## change this for csd3 
pop_cosmos_path = "/home/jjg65/rds/hpc-work/pop-cosmos"
models_path = pop_cosmos_path + "/trained_models/"
blast_models = "photulator_models/Blast/model_4x128_alsing20_"

# catalog trained in thorp et al. 2025
catalog_model = torch.load(models_path + "catalogueModelT25.pt", map_location=device, weights_only=False)
catalog_model = catalog_model.to(device)

dat = pd.read_csv('/home/jjg65/ztf_mcmc/blast_global_data_merged_cuts_SNR_score_z_unextONLY.csv')    # dust corrected
dat = dat[dat['ebv_MW'] > 0]       # this is where all fluxes are 0   # it is possible to have the mv_ebv value from the ztf_snia, but regardless there is no photometry :/ 
dat = dat[(dat['redshift'] > 0.023) & (dat['redshift'] < 0.06)]

inds_narrow = np.where((dat['redshift_err'] > 0.001).values)[0]
inds_fixed = np.where((dat['redshift_err'] < 0.001).values)[0]
inds = np.arange(330)
# inds = np.arange(len(dat))        # change this for test case
# inds = np.arange(3)

# %%
# Set flux to 0 & fluxerrs to huge for unavailable bands:
all_flux_bands = np.array(('flux_GALEX_FUV', 'flux_GALEX_NUV', 'flux_PanSTARRS_g',
       'flux_PanSTARRS_r', 'flux_PanSTARRS_i', 'flux_PanSTARRS_z',
       'flux_PanSTARRS_y', 'flux_SDSS_u', 'flux_SDSS_g', 'flux_SDSS_r',
       'flux_SDSS_i', 'flux_SDSS_z', 'flux_DES_g', 'flux_DES_r', 'flux_DES_i',
       'flux_DES_z', 'flux_DES_Y', 'flux_2MASS_J', 'flux_2MASS_H',
       'flux_2MASS_K', 'flux_WISE_W1', 'flux_WISE_W2', 'flux_WISE_W3',
       'flux_WISE_W4'))
all_flux_bands = np.array([all_flux_bands[i] + '_unext' for i in range(len(all_flux_bands))])
all_fluxerr_bands = [all_flux_bands[i].replace('flux', 'fluxerr') for i in range(len(all_flux_bands))]

df_bool = (dat.iloc[inds][all_flux_bands] > 0)
df_bool2 = (dat.iloc[inds][all_fluxerr_bands] > 0)
row_idx, col_idx = np.where(df_bool)
row_counts = df_bool.sum(axis=1)

all_fluxes_nan = torch.from_numpy(dat.iloc[inds][all_flux_bands][df_bool].values).to(dtype=torch.float64, device=device)
all_fluxes_nan[torch.isnan(all_fluxes_nan)] = 0                    
all_fluxerrs_nan = torch.from_numpy(dat.iloc[inds][all_fluxerr_bands][df_bool2].values).to(dtype=torch.float64, device=device)
all_fluxerrs_nan[torch.isnan(all_fluxerrs_nan)] = torch.inf

redshift = torch.tensor(dat['redshift'].iloc[inds].values, dtype=torch.float64, device=device)
redshifterr = torch.tensor(dat['redshift_err'].iloc[inds].values, dtype=torch.float64, device=device)
redshifterr[redshifterr < 0.001] = torch.nan
zsource = dat['z_source'].iloc[inds].values
ztfname = dat['ztf_name'].iloc[inds].values
iauname = dat['iau_name'].iloc[inds].values

fixed = torch.isnan(redshifterr).to(device=device)
nfixed = len(torch.where(fixed == True)[0])
nnarrow = len(torch.where(fixed == False)[0])
print('fixed =', nfixed)
print('narrow =', nnarrow)

# %%
# bands to put into photulator (in log_prob function)
bands = [['galex_' + f'{f}UV' for f in 'FN'] + ['panstarrs_' + f'{f}' for f in 'grizy'] 
         + ['sdss_' + f'{f}0' for f in 'ugriz'] + ['decam_' + f'{f}' for f in 'grizY'] 
         + ['twomass_' + f'{f}' for f in ['J', 'H', 'Ks']] + ['wise_' + f'w{f}' for f in '1234']]
bands = np.array(bands[0])
max_photulators = PhotulatorModelStack('', [models_path + blast_models + f'{b}_cpu.pt' for b in bands], device=device)    

## from model draws, get prior fluxes for valid bands 
prior_fluxes_all_no_decam_u = torch.load('/home/jjg65/ztf_mcmc/prior_fluxes_all_no_decam_u_mJy.pt', map_location=device, weights_only=False)
prior_theta = torch.load('/home/jjg65/ztf_mcmc/prior_theta.pt', map_location=device, weights_only=False)
# prior_fluxes_flat = prior_fluxes_all_no_decam_u[:, col_idx]
# prior_fluxes = torch.split(prior_fluxes_flat, row_counts.values.tolist(), dim=1)
    # order: 
    # 0) galex: fuv, nuv (0-1)
    # 1) panstarrs: grizy (2-6)
    # 2) sdss: ugriz (7-11)
    # 3) decam: ugrizY (12-17)
    # 4) 2mass: jhk_s (18-20)
    # 5) wise: 1234 (21-24)

print('calculating x2...') 
# %%
all_argmins = []

for i in range(all_fluxes_nan.shape[0]):
    prior_fluxes = prior_fluxes_all_no_decam_u[:, df_bool.iloc[i].values]
    diff = all_fluxes_nan[i][torch.nonzero(all_fluxes_nan[i]).flatten()] - prior_fluxes
    x22 = torch.sum((diff / all_fluxerrs_nan[i][~torch.isinf(all_fluxerrs_nan[i])])**2, dim=1)
    all_argmins.append(torch.argmin(x22))
argmins = torch.stack(all_argmins).to(device)

# diffs = all_fluxes_nan.flatten()[torch.nonzero(all_fluxes_nan.flatten()).flatten()] - prior_fluxes_flat
# sqs = (diffs / all_fluxerrs_nan[~torch.isinf(all_fluxerrs_nan)])**2
# splits = torch.split(sqs, row_counts.tolist(), dim=1)
# x2 = torch.stack([torch.sum(s, dim=1) for s in splits])

# argmins = torch.argmin(x2, dim=1)
theta0 = prior_theta[argmins].to(dtype=torch.float64)
theta0[:, -1] = redshift
phi0 = catalog_model.theta2phi(theta0)


# %%
def log_prob_batch_phi16(phi, flux, fluxerr, red, rederr, photulators, alpha=2.0):

    phi = phi.to(torch.float32)
    theta = catalog_model.phi2theta(phi)

    m = photulators.magnitudes(
        catalog_model.emulator_parameter_transform(theta)[..., 1:],
        torch.unsqueeze(theta[..., 0], -1)
    )
    f = 10 ** ((m - 22.5) / -2.5)
    f = f * 3.631  # µJy
    # emission line corrections ... fill in later

    ll = torch.sum(-0.5 * ((flux - f) / fluxerr) ** 2., dim=-1)
    lz = alpha * -0.5 * ((red - theta[..., -1]) / rederr) ** 2.

    lp = catalog_model.population_model.log_prob(phi.view(-1, 16)).view(ll.shape)

    return lp + ll + lz


# %%
sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))
def log_prob_batch_phi15(phi_or_phi15, flux, fluxerr, red, rederr, photulators, alpha=2.0): # rederr is not needed
    
    phi15 = phi_or_phi15 
    red_phi = (red - catalog_model.lower[-1]) / (catalog_model.range[-1])
    # red_phi = np.sqrt(2.) * torch.erfinv(2.0 * red_phi - 1.0)            
    # red_phi = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2.0 * red_phi - 1.0)
    red_phi = sqrt2 * torch.erfinv(2.0 * red_phi - 1.0)

    if flux.dim() > 2:  # for mcmc fitting
        red_phi = red_phi.view(1, -1, 1).expand(phi15.shape[0], -1, -1)   # (n_walkers, n_gal, 1)
        phi = torch.cat((phi15, red_phi), dim=-1).to(torch.float32)       # (n_walkers, n_gal, 16)
    
    else: 
        phi = torch.cat((phi15, red_phi.view(-1, 1)), dim=1).to(torch.float32)
    
    theta = catalog_model.phi2theta(phi)

    m = photulators.magnitudes(catalog_model.emulator_parameter_transform(theta)[...,1:], torch.unsqueeze(theta[...,0], -1)) 
    f = 10 ** ((m - 22.5) / -2.5)   
    f = f * 3.631           # in microjanskys (uJy)
    # emission line corrections ... fill in later

    ll = torch.sum(-0.5*((flux - f)/fluxerr)**2., dim=-1)

    lz = torch.zeros(ll.shape, dtype=torch.float64, device=device)

    lp = catalog_model.population_model.log_prob(phi.view(-1, 16)).view(ll.shape)
    
    return lp + ll + lz


# %%
# phi0_15 = np.copy(phi0.detach().numpy())[fixed, :-1]
phi0_15 = phi0.detach()[fixed, :-1]
# phi0_16 = np.copy(phi0.detach().numpy())[~fixed, :]
phi0_16 = phi0.detach()[~fixed, :]

# pmaps15 = torch.nn.Parameter(torch.from_numpy(phi0_15)).to(device)
pmaps15 = torch.nn.Parameter(phi0_15)
# pmaps16 = torch.nn.Parameter(torch.from_numpy(phi0_16)).to(device)
pmaps16 = torch.nn.Parameter(phi0_16)
    
log_prob_func15 = lambda phi: log_prob_batch_phi15(
    phi, 
    all_fluxes_nan[fixed], all_fluxerrs_nan[fixed], 
    redshift[fixed], redshifterr[fixed],
    max_photulators, 
    alpha=2.0)
    
log_prob_func16 = lambda phi: log_prob_batch_phi16(
    phi, 
    all_fluxes_nan[~fixed], all_fluxerrs_nan[~fixed], 
    redshift[~fixed], redshifterr[~fixed],
    max_photulators, 
    alpha=2.0)

optimizer = torch.optim.Adam([pmaps15, pmaps16], lr=1e-3)

print('optimizing...') 
for _ in tqdm(range(200)):
    optimizer.zero_grad()
    loss = -log_prob_func15(pmaps15).sum() \
       -log_prob_func16(pmaps16).sum()
    loss.backward()
    optimizer.step()

base_dir = '/home/jjg65/rds/hpc-work/pmaps/'
path15 = 'pmaps15_I.pt'
path16 = 'pmaps16_I.pt'
torch.save(pmaps15, os.path.join(base_dir + path15))
torch.save(pmaps16, os.path.join(base_dir + path16))

# %%
def mcmc(pmaps, log_prob, n_params, n_gal, n_walkers=256, n_steps=1000):  
    """pmaps: Tensor, phi_map estimate to initialize MCMC
    log_prob: function
    n_params: 15 or 16
    n_gal: int, number of galaxies in fixed or narrow redshift
    """
    walkers1 = torch.randn(n_walkers, n_gal, n_params, device=device)*0.1 + pmaps
    walkers2 = torch.randn(n_walkers, n_gal, n_params, device=device)*0.1 + pmaps
    
    if log_prob == log_prob_batch_phi15: 
        cond = fixed
        ztype = 'fixed'
        num = nfixed
    else:    # log_prob == log_prob_batch_phi16:
        cond = ~fixed
        ztype = 'narrow'
        num = nnarrow
    
    f_obs_broadcast = all_fluxes_nan[cond].broadcast_to(n_walkers, n_gal, 24) 
    f_err_broadcast = all_fluxerrs_nan[cond].broadcast_to(n_walkers, n_gal, 24)
    z_obs = redshift[cond]
    z_err = redshifterr[cond]
        
    log_prob_func = lambda phi: log_prob(phi, f_obs_broadcast, f_err_broadcast,
                                                      z_obs, z_err, max_photulators, alpha=2.0) 
    with torch.no_grad():
        mcmc_phi = sample_batch(log_prob_func, n_steps, [walkers1, walkers2], progress=True, device=device)
            
    if log_prob == log_prob_batch_phi15: 
        red_phi = (z_obs - catalog_model.lower[-1]) / (catalog_model.range[-1])
        # red_phi = np.sqrt(2.) * torch.erfinv(2.0 * red_phi - 1.0)   
        # red_phi = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2.0 * red_phi - 1.0)
        red_phi = sqrt2 * torch.erfinv(2.0 * red_phi - 1.0)
        fill_red = red_phi.broadcast_to(mcmc_phi.reshape(-1, n_gal, 15).shape[0], red_phi.shape[0])
        mcmc_phi = torch.cat((mcmc_phi.reshape(-1, n_gal, 15), fill_red.unsqueeze(-1)), dim=-1)

    # mcmc_theta = catalog_model.phi2theta(mcmc_phi)[int(mcmc_phi.shape[0]/2):].reshape(-1, n_gal, 16).detach().numpy()
    mcmc_theta = catalog_model.phi2theta(mcmc_phi)[int(mcmc_phi.shape[0]/2):].reshape(-1, n_gal, 16).detach()
    base_dir = '/home/jjg65/rds/hpc-work/ztf_mcmc_chains/'
    # base_dir = '/home/jjg65/rds/hpc-work/ztf_mcmc_chains/test'
    # base_dir = '/home/jjg65/rds/hpc-work/ztf_mcmc_chains/'
    path = f'mcmc_theta_{ztype}_{num}_I.pt'

    torch.save(mcmc_theta, os.path.join(base_dir + path))
#     np.save(os.path.join(base_dir, path), mcmc_theta)
    
    # note make sure to get rid of the last -512 for posterior (bug in affine)


# %%
mcmc(pmaps15, log_prob_batch_phi15, n_params=15, n_gal=int(fixed.sum()), n_walkers=256, n_steps=1000)
mcmc(pmaps16, log_prob_batch_phi16, n_params=16, n_gal=int((~fixed).sum()), n_walkers=256, n_steps=1000)

