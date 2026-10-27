import numpy as np
import torch

def sq_exp(X1, X2, params):
    """
    Compute covariance matrix K(X1, X2) using squared-exponential kernel. 
    Inputs: 
        X1: set 1 of input points; shape (n, D)
        X2: set 2 of input points; shape (m, D)
        params:
            - ell: lengthscales, one per feature or shared; array of shape (D,)
            - sigma_f: signal variance, overall amplitude; float
    Outputs: 
        K: shape (n, m) 
    """
    diff = (X1[:, None, :] - X2[None, :, :]) / params["ell"]      # (n, n, d)
    sqdist = torch.sum(diff**2, dim=-1)                 # (n, n)    
    K = params["sigma_f"]**2 * torch.exp(-0.5 * sqdist)           # (n, n)
    
    # if using noisy data, add sigma_n^2 + I
    return K

def sq_exp_batch(X1, X2, params):
    """Batched squared-exponential kernel. Same as sq_exp with a leading realization axis.
    Inputs:
        X1: (K, n, D)
        X2: (K, m, D)
        params:
            - sigma_f: overall amplitude, one per realization; shape (K,)
            - ell: lengthscales, one set per realization; shape (K, D)
    Outputs:
        K: (K, n, m)
    """
    d = (X1[:, :, None, :] - X2[:, None, :, :]) / params["ell"][:, None, None, :]
    return (params["sigma_f"] ** 2)[:, None, None] * torch.exp(-0.5 * (d ** 2).sum(-1))


