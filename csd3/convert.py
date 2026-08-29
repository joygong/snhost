import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

pop_cosmos_path = "/home/jjg65/rds/hpc-work/pop-cosmos"
models_path = pop_cosmos_path + "/trained_models/"
blast_models = "photulator_models/Blast/model_4x128_alsing20_"

# catalog trained in thorp et al. 2025
catalog_model = torch.load(models_path + "catalogueModelT25.pt", map_location=device, weights_only=False)

for i in tqdm(np.arange(3, 5)):
# for i in tqdm(range(5)): 
    mcmc_phi = torch.load(f'mcmc_phi_fixed_479_{i}.pt', map_location=device)
    print(mcmc_phi.shape[0], int(mcmc_phi.shape[0]/2))
    mcmc_theta = catalog_model.phi2theta(mcmc_phi)
    print('mcmc_theta shape =', mcmc_theta.shape)
    mcmc_theta_half = mcmc_theta[int(mcmc_theta.shape[0]/2):].cpu().detach().numpy()
    print('half shape =', mcmc_theta_half.shape)
    np.save(f'mcmc_theta_fixed_479_{i}.npy', mcmc_theta_half)
