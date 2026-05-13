import numpy as np
import torch 

dtype = torch.float32

# def train(kernel, X_train, y_train, params, y_train_noise=None, jitter=1e-13): 
#     """
#     Calculate predictive mean and variance. 
#     Inputs: 
#         kernel: kernel function to use  
#         X_train: training input points; shape (n, D)
#         y_train: training target points; shape (n, 1)
#         X_test: testing input points; shape (m, D)
#         params: 
#             - ell: lengthscales, one per feature or shared; array of shape (D,)
#             - sigma_f: signal variance, overall amplitude; float
#         y_noise: optional, sigma_n^2, per-point variance in targets, points assumed to be independent; shape (n,)
#         jitter: added for numerical stability; float 
#     Outputs: 
#         f_pred: predictive mean; shape (m, 1)
#         cov_pred: predictive (co)variance; shape (m, m)
#         p: log marginal likelihood; shape (1, 1)
#     """

def predict(kernel, X_train, y_train, X_test, params, y_train_noise=None, jitter=1e-13):
    """
    Calculate predictive mean and variance. 
    Inputs: 
        kernel: kernel function to use  
        X_train: training input points; shape (n, D)
        y_train: training target points; shape (n, 1)
        X_test: testing input points; shape (m, D)
        params: 
            - ell: lengthscales, one per feature or shared; array of shape (D,)
            - sigma_f: signal variance, overall amplitude; float
        y_noise: optional, sigma_n^2, per-point variance in targets, points assumed to be independent; shape (n,)
        jitter: added for numerical stability; float 
    Outputs: 
        f_pred: predictive mean; shape (m, 1)
        cov_pred: predictive (co)variance; shape (m, m)
        p: log marginal likelihood; shape (1, 1)
    """
    # Helper variables 
    n = X_train.shape[0]
    I = torch.eye(n)

    # Reshape y_noise to diagonal matrix. Compute training covariance. 
    if y_train_noise is not None: 
        y_train_noise = torch.diag(y_train_noise)
        K = kernel(X_train, X_train, params) + y_train_noise * I 
    else: 
        K = kernel(X_train, X_train, params) + jitter * I

    # Compute (rest of) covariance blocks      
    Ks = kernel(X_train, X_test, params)                            # doesn't need jitter, b/c no decomposition 
    Kss = kernel(X_test, X_test, params)                            # doesn't need jitter, b/c no decomposition 

    # Cholesky
    L = torch.linalg.cholesky(K, upper=False)                       # (n, n), lower triangular

    # Solve for alpha
    b = torch.linalg.solve_triangular(L, y_train, upper=False)      # (n, 1), L = lower triangular; only lower part is accessed
    alpha = torch.linalg.solve_triangular(L.T, b, upper=True)       # (n, 1), L.T = upper triangular; only upper part is accessed

    # Predictive mean (m, n) @ (n, 1) -> (m, 1)
    f_pred = Ks.T @ alpha                                           # target function mean; (m, 1). mean prior assumed to be 0. 

    # Predictive variance (m, m) - (m, n) @ (n, m) -> (m, m)
    v = torch.linalg.solve_triangular(L, Ks, upper=False)
    cov_pred = Kss - v.T @ v                                        # target function variance; (m, m). mean prior assumed to be 0. 

    # Log marginal likelihood log p(y|X). (m,)
    t1 = -0.5 * y_train.T @ alpha 
    t2 = - torch.sum(torch.log(torch.diagonal(L)))
    t3 = -0.5 * n * torch.log(torch.tensor(2 * torch.pi))
    p = t1 + t2 + t3                                                # log marginal likelihood (marginalized over all f); (1, 1). 

    return f_pred, cov_pred, p 
 
def optimize_GP(raw_params, kernel, X_train, base_params, epochs=2000): 
    """Optimize M0, sigma_int, & GP hyperparameters (overall variance amplitude sigma_f, lengthscales ells). 
    Inputs: 
        raw_params: starting parameters for optimization; torch.nn.Parameter(torch.tensor())
            [raw_M0, raw_log_sigma_int, raw_log_sigma_f, raw_log_ell_1, ..., raw_log_ell_D]
            - M0: absolute magnitude 
            - sigma_int: intrinsic scatter
            - sigma_f: overall variance amplitude in kernel
            - ell: lengthscale for each input 
        kernel: kernel to evaluate GP 
        X_train: training input data
        base_params: parameters that do not change; shape (3, n)
            [mu_th, mB, y_noise_meas] 
        epochs: # epochs for which to optimize
    """
    def _log_marginal_likelihood(raw_params, kernel, X_train, base_params):

        """
        Calculate log marginal likelihood for a set of M0, sigma_int, sigma_f, ells. 
        Inputs: 
            raw_params: logs are inputted for parameters required to be positive. 
                [raw_M0, raw_log_sigma_int, raw_log_sigma_f, raw_log_ell_1, ..., raw_log_ell_D]
                - M0: absolute magnitude 
                - sigma_int: intrinsic scatter
                - sigma_f: overall variance amplitude in kernel
                - ell: lengthscale for each input 
            base_params: (3, n)
                [mu_th, mB, y_noise_meas] <- for now, input 0 for noise_meas. figure out other none case later 
        Returns: 
            p: log marginal likelihood 
        """
        # Helper variables
        n, D = X_train.shape
        mu_th, mB, y_noise_meas = base_params.T
        I = torch.eye(n)

        # Extract parameters
        M0 = raw_params[0]
        sigma_int = torch.exp(raw_params[1])                            # exponentiate to guarantee positive values. sigma_int scaled same as y_noise_meas. 
        sigma_f = torch.exp(raw_params[2])                      
        ell = torch.exp(raw_params[3:3+D]) 

        # y = y(M0)
        y = (mu_th - (mB - M0)).reshape(-1, 1).to(dtype)                                          # (n,)

        # y_noise standardized by y_std, which is independent of M0 
        y_noise_tot = y_noise_meas + sigma_int**2
        y_noise_tot = y_noise_tot.to(dtype)

        # Kernel
        params = {}
        params["sigma_f"], params["ell"] = sigma_f, ell
        K = kernel(X_train, X_train, params)
        Ky = K + torch.diag(y_noise_tot) * I
        
        # Cholesky
        L = torch.linalg.cholesky(Ky, upper=False)                       # (n, n), lower triangular

        # Solve for alpha
        b = torch.linalg.solve_triangular(L, y, upper=False)            # (n, 1), L = lower triangular; only lower part is accessed
        alpha = torch.linalg.solve_triangular(L.T, b, upper=True)       # (n, 1), L.T = upper triangular; only upper part is accessed

        # Log marginal likelihood log p(y|X). (m,)
        t1 = -0.5 * y.T @ alpha 
        t2 = - torch.sum(torch.log(torch.diagonal(L)))
        t3 = -0.5 * n * torch.log(torch.tensor(2 * torch.pi))
        p = t1 + t2 + t3                                                # log marginal likelihood (marginalized over all f); (1, 1). 

        return p 

    optimizer = torch.optim.Adam([raw_params], lr=0.05)

    for step in range(epochs):
        optimizer.zero_grad()
        loss = -_log_marginal_likelihood(raw_params, kernel, X_train, base_params)        # start as y_meas_var = sig_mB**2 + sig_mu_th**2
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(step, loss.item())

    M0_fit = raw_params[0].item()                                   # this doesn't change ? 
    sigma_int_fit = torch.exp(raw_params[1]).item()                 # this still needs to be rescaled
    sigma_f_fit = torch.exp(raw_params[2]).item()
    ell_fit = torch.exp(raw_params[3:]).detach().cpu().numpy()

    # Pretty printing
    ell_labels = ["x1", "c", "M", "Z", "sSFR", "dust2"]

    print(f"M0         = {M0_fit:.4f}")
    print(f"sigma_int  = {sigma_int_fit:.4f}")
    print(f"sigma_f    = {sigma_f_fit:.4f}")
    print("ell:")

    for label, val in zip(ell_labels, ell_fit):
        print(f"  {label:<6} = {val:.4f}")

    return {'M0': M0_fit, 
            'sigma_int': sigma_int_fit, 
            'sigma_f': sigma_f_fit, 
            'ell': ell_fit}

def xtest(input_var, kernel, X_train, y_train, **predict_kwargs):
    """
    Generate and predict mean, variance, and log marginal likelihood for test points. 
    Inputs: 
        input_var: SN or galaxy parameter to sample over. Other variables held fixed. 
        kernel: kernel functon to use
        X_train: training input points; shape (n, D)
        y_train: training target points; shape (n, 1)
        **predict_kwargs: params=params, y_train_noise=y_noise, jitter=1e-13 from predict function. 
    Outputs: 
        f_pred: predictive mean; shape (m,)
        cov_pred: predictive variance; shape (m,)
        p: log marginal likelihood; shape (m,)
        X_test: test points 
    """
    X_test = torch.median(X_train, dim=0)[0].unsqueeze(0).repeat(100, 1)
    
    i = input_indices[input_var]
    grid = torch.linspace(X_train[:, i].min(), X_train[:, i].max(), 100)
    X_test[:, i] = grid

    f_pred, cov_pred, p = predict(kernel, X_train, y_train, X_test, **predict_kwargs)

    diag_var = torch.diagonal(cov_pred)
    if torch.any(diag_var < 0):
        bad_vals = diag_var[diag_var < 0]
        raise ValueError(f"Predictive variances contain negative diagonal entries: {bad_vals}")
    
    return f_pred, cov_pred, p, X_test
