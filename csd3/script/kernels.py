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


