import torch
import tqdm
import numpy as np

# fixed = np.load('mcmc_theta_fixed_479_1.npy')
fixed = np.load('mcmc_theta_fixed_479_2.npy')
n_chunks = 5
total_rows = fixed.shape[1]
chunk_size = int(np.ceil(total_rows / n_chunks))

for i in range(n_chunks):
    start = i * chunk_size
    end = min((i+1) * chunk_size, total_rows)
    # np.save(f"mcmc_theta_fixed_479_1_{i}.npy", fixed[:, start:end, :])
    np.save(f"mcmc_theta_fixed_479_2_{i}.npy", fixed[:, start:end, :])
    # np.save(fixed[:, start:end, :], f"mcmc_theta_fixed_479_1_{i}.npy")
#     np.save(fixed[:, start:end, :], f"mcmc_theta_fixed_479_2_{i}.npy")

