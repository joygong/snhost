import numpy as np
from tqdm import tqdm 
from tqdm import trange
import torch
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from flowfusion.diffusion import VESDE, VPSDE, MLP, ScoreModel, denoising_score_matching


def __init__(X, sde, n_cond): 
    """ 
    Initialize score model. 
    
    Parameters: 
        X: concatenated tensor of conditionals and target; (633, 7)
        sde: VESDE() or VPSDE()

    Returns:
        score_model: untrained score model
        n_cond: conditional dimensions
        n_dims: target dimensions  
    """
    n_dims = X.shape[1] - n_cond

    score_network = MLP(n_dimensions=n_dims, n_conditionals=n_cond, units=[128, 128, 128])     
    score_model = ScoreModel(model=score_network, sde=sde) 

    return score_model, n_cond, n_dims

def train(score_model, X_train, n_cond, ratio_t, ratio_v, optimize_M0=True, M_0=None, plot=True, epochs=200):  
    """
    Train a diffusion model to optimize weights, biases, and M_0. 
    
    Parameters: 
        score_model: initialized score model specificed by score_network (architecture) & sde type 
        X_train: conditional (X_train[:, :-1]) + target base data (y_base = X[:, -1] = mu_cat - mB_cat)
        ratio_t: training ratio, default 0.8
        ratio_v: validation ratio, default 0.2 
        optimize_M0: option to optimize for M0
        M_0: initialization of M0 
        plot: option to plot training and validation losses
        epochs: # epochs for which to train model 

    Returns: 
        score_model: optimized score_model 
        M0: optimized M0
        losses: training losses
        test_losses: validation losses
    """
    losses = []
    test_losses = []
    steps_per_epoch = []

    g = torch.Generator().manual_seed(42)

    print('training...')
    for bs, lr in zip ([32, 64, 128, 256], [1e-03, 1e-04, 1e-05, 1e-06]):       # bs and lr linked 
        print('batch size:', bs)
        print('learning rate:', lr)
        
        M0 = torch.nn.Parameter(M_0)
        
        if optimize_M0: 
            optimizer = Adam(list(score_model.model.parameters()) + [M0], lr=lr)
        else: 
            optimizer = Adam(score_model.model.parameters(), lr=lr)
        
        n_train, n_test = random_split(X_train, (ratio_t, ratio_v), generator=g)

        data_loader = DataLoader(n_train, shuffle=True, batch_size=bs)
        for _ in trange(epochs):
            c = 0                           # mini-batch 
            for n in data_loader:           # loop over each mini batch in training data
                optimizer.zero_grad()   
                loss = denoising_score_matching(score_model, n[:,n_cond:] + M0, conditional=n[:,:n_cond])    # add only to the targets, not conditionals, s.t. y = mu - (mB - M0)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
                c += 1
            test_loss = denoising_score_matching(score_model, n_test.dataset[n_test.indices][:,n_cond:] + M0, conditional=n_test.dataset[n_test.indices][:, :n_cond])
            test_losses.append(test_loss.detach().numpy())
            steps_per_epoch.append(c)
        print('bs:', bs)
        print('lr:', lr)
        print('M0_fit:', M0)

    print('done')
 
    if plot: 
        plt.plot(1+np.arange(len(losses)), losses, color='k', linewidth=0.2, label='training loss')             # 4*epochs
        plt.plot(np.cumsum(steps_per_epoch), test_losses, color='r', linewidth=1.0, label='validation loss')    # (ceil(len(n_train) / 32) + ceil(len(n_train) / 64 + ... etc.) * epochs = 30*epochs
        plt.xlabel('step')
        plt.ylabel('score-matching loss')
        plt.legend(frameon=False)

    return score_model, M0.detach(), losses, test_losses

def predict_SN(score_model, X, n_cond, n_samples, M0, steps=100, batch_size=32): 
    """
    Predict mu - mB for each supernova. 

    Parameters: 
        score_model: trained score model  
        X: full concatenated tensor = conditional (X_train[:, :-1]) + target base data (y_base = X[:, -1] = mu_cat - mB_cat)
        n_samples: # samples to generate for each SN 
        M0: optimized M0 fit for model
        steps: # steps for diffusion model to denoise from T -> 0 
        batch_size: batch size to compute SNe predictions; caution against setting batch_size=n_sn (MEMORY OVERLOAD!)

    Returns: 
        y_pred_SN: predictions for each SN; (n_sn, n_samples) 
        y_pred_SN_med: median predictions for each SN; (n_sn,)
        y_pred_SN_std: std predictions for each SN; (n_sn,)
    """
    X_cond = X[:, :n_cond]
    n_sn = X_cond.shape[0]
    M0 = torch.as_tensor(M0, device=X_cond.device, dtype=X_cond.dtype)

    y_pred_SN = torch.empty((n_sn, n_samples), device=X_cond.device, dtype=X_cond.dtype)

    for start in tqdm(range(0, n_sn, batch_size)):                                          # batches to not overload memory
        stop = min(start + batch_size, n_sn)
        cond_batch = X_cond[start:stop]
        cond_rep = cond_batch[:, None, :].expand(-1, n_samples, -1).reshape(-1, n_cond)     # n_samples x SN[i], for however many SN in batch 

        y_pred = score_model.sample_sde((cond_rep.shape[0], 1), conditional=cond_rep, steps=steps)
        y_pred_SN[start:stop] = y_pred.reshape(stop - start, n_samples) - M0

    y_pred_SN_med = torch.median(y_pred_SN, dim=1)[0]
    y_pred_SN_std = torch.std(y_pred_SN, dim=1)

    return y_pred_SN, y_pred_SN_med, y_pred_SN_std
