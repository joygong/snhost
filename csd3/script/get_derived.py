import numpy as np
import torch
import pickle
import astropy
from astropy.cosmology.realizations import Planck18
from pop_cosmos.utils import compute_derived_quantities, compute_mass_remaining

device = "cpu"
pop_cosmos_path = "/home/jjg65/rds/hpc-work/pop-cosmos"
models_path = pop_cosmos_path + "/trained_models/"
blast_models = "photulator_models/Blast/model_4x128_alsing20_"
catalog_model = torch.load(models_path + "catalogueModelT25.pt", map_location=device, weights_only=False)
mass_remaining_emulator = torch.load(models_path + "stellar_mass_emulator.pt", map_location=device, weights_only=False)
mass_theta_shift, mass_theta_scale = pickle.load(open(models_path + "stellar_mass_parameter_shift-scale.pkl", "rb"))
mass_theta_shift = torch.from_numpy(mass_theta_shift).to(device)
mass_theta_scale = torch.from_numpy(mass_theta_scale).to(device)

print("imports done!") 

def get_derived(file_name, chunk_num): 
    ### for the fixed redshift case
    chains = torch.load(f"{file_name}", map_location=device)
    chains = chains[:-1, :, :, :]
    chains = chains.reshape(-1, chains.shape[2], 16) 
    
    a = [compute_derived_quantities(chains[:,i,:]) for i in range(chains.shape[1])]
    a = torch.tensor(np.array(a)).to(device)
    log10M_formed = a[:, 0, :]
    mw_age = a[:, 1, :]
    log10SFR = a[:, 2, :]
    log10sSFR = a[:, 3, :]
    
    torch.save(log10M_formed, f"2000_log10M_formed_fixed_{chunk_num}.pt")
    torch.save(mw_age, f"2000_mw_age_fixed_{chunk_num}.pt")
    torch.save(log10SFR, f"2000_log10SFR_fixed_{chunk_num}.pt")
    torch.save(log10sSFR, f"2000_log10sSFR_fixed_{chunk_num}.pt")
    
    b_01 = [compute_mass_remaining(log10M_formed[i], log10sSFR[i], chains[:, i, :], mass_theta_shift, mass_theta_scale, mass_remaining_emulator) for i in range(chains.shape[1])]
    b_01_tensor = torch.zeros((len(b_01), 3, 1023488))

    for i in range(len(b_01)): 
        for j in range(3): 
            b_01_tensor[i][j] = b_01[i][j].detach()

    log10M_remain_01 = b_01_tensor[:, 0, :]
    log10sSFR_01 = b_01_tensor[:, 1, :]
    M_frac_01 = b_01_tensor[:, 2, :]

    torch.save(log10M_remain_01, f"2000_log10M_remain_fixed_{chunk_num}.pt")
    torch.save(log10sSFR_01, f"2000_log10sSFR_fixed_{chunk_num}.pt")
    torch.save(M_frac_01, f"2000_M_frac_fixed_{chunk_num}.pt")

get_derived("2000_mcmc_theta16_fixed_479_01_2.pt" , '012') 
print('012 done!')
get_derived("2000_mcmc_theta16_fixed_479_02_1.pt", '021') 
print('021 done!')
get_derived("2000_mcmc_theta16_fixed_479_02_2.pt", '022') 
print('022 done!')
