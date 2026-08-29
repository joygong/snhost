import torch
from tqdm import tqdm
import numpy as np

fixed = torch.load('og_mcmc_phi_fixed_479.pt', map_location='cpu')
n_chunks = 5
total_rows = fixed.shape[1]
print(total_rows)
chunk_size = int(np.ceil(total_rows / n_chunks))

for i in tqdm(range(n_chunks)):
    start = i * chunk_size
    end = min((i+1) * chunk_size, total_rows)
    torch.save(fixed[:, start:end, :], f"mcmc_phi_fixed_479_{i}.pt")

