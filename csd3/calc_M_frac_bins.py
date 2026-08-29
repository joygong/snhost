import numpy as np 
import pandas as pd
import torch
from pop_cosmos.utils import specific_star_formation_rate

device = "cpu"

def calc_M_frac_bins(file_name, save_file_name):
    chains = torch.load(f"/home/jjg65/rds/hpc-work/ztf_mcmc_chains/{file_name}", map_location=device)
    chains = chains[:-1, :, :, :]
    chains = chains.reshape(-1, chains.shape[2], 16) 
    print(chains.shape)
    print('# galaxies:', chains.shape[1])
    # log10sSFR, M_frac_bins, t_edge_bins = specific_star_formation_rate(thetas[:,2:8], z_cat, return_bins=True)
    a = [specific_star_formation_rate(chains[:,i,2:8], chains[:,i,-1], return_bins=True) for i in range(chains.shape[1])]
    M_frac_bins = np.array([a[i][1] for i in range(chains.shape[1])])
    M_frac_bins = torch.tensor(M_frac_bins).to(dtype=torch.float32)
    torch.save(M_frac_bins, f"/home/jjg65/rds/hpc-work/ztf_mcmc_chains/{save_file_name}")

    return M_frac_bins.shape, M_frac_bins

print('calc fixed_011...')
calc_M_frac_bins("2000_mcmc_theta16_fixed_479_01_1.pt", "2000_M_frac_bins_011.pt")
print('DONE!')
print('calc fixed_012...')
calc_M_frac_bins("2000_mcmc_theta16_fixed_479_01_2.pt", "2000_M_frac_bins_012.pt")
print('DONE!')
print('calc fixed_021...')
calc_M_frac_bins("2000_mcmc_theta16_fixed_479_02_1.pt", "2000_M_frac_bins_021.pt")
print('DONE!')
print('calc fixed_022...')
calc_M_frac_bins("2000_mcmc_theta16_fixed_479_02_2.pt", "2000_M_frac_bins_022.pt")
print('DONE!')

