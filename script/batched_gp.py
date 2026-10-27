import numpy as np
import torch 

dtype = torch.float32

def to_raw(d, *, dtype=torch.float64, device=None):
    """Param dict -> raw vector for the optimizer.

    Inputs:
        d: {'M0', 'sigma_int', 'sigma_f', 'ell'}, as returned by optimize_GP
    Outputs:
        raw: [M0, log sigma_int, log sigma_f, log ell_1, ..., log ell_D]; shape (3 + D,)
    """
    return torch.tensor(
        [float(d['M0']),
         float(np.log(d['sigma_int'])),
         float(np.log(d['sigma_f']))]
        + np.log(np.asarray(d['ell'], dtype=np.float64)).tolist(),
        dtype=dtype, device=device,
    )

def from_raw(row):
    """Raw vector -> param dict, same format as optimize_GP returns.

    Inputs:
        row: [M0, log sigma_int, log sigma_f, log ell...]; shape (3 + D,)
    Outputs:
        dict with M0, sigma_int, sigma_f as floats and ell as a numpy array
    """
    return {'M0':        row[0].item(),
            'sigma_int': torch.exp(row[1]).item(),
            'sigma_f':   torch.exp(row[2]).item(),
            'ell':       torch.exp(row[3:]).detach().cpu().numpy()}

def stack_params(param_list, *, dtype=torch.float64, device=None):
    """List of param dicts -> batched kernel params for predict_batch.

    sigma_int is NOT returned: it is not a kernel parameter. It belongs in the
    training noise, which the caller builds as y_noise_e + sigma_int**2.

    Inputs:
        param_list: list of K dicts, as returned by optimize_GP_batch
    Outputs:
        P: {'sigma_f': (K,), 'ell': (K, D)}
        M0: (K,)
    """
    P = {"sigma_f": torch.tensor([p['sigma_f'] for p in param_list],
                                 dtype=dtype, device=device),
         "ell": torch.tensor(np.array([p['ell'] for p in param_list]),
                             dtype=dtype, device=device)}
    M0 = torch.tensor([p['M0'] for p in param_list], dtype=dtype, device=device)
    return P, M0
    
def optimize_GP_batch(raw_params, kernel, X_train, base_params, epochs=2000, lr=0.05, gamma=0.998):
    """Batched optimize_GP: fits K independent GPs at once.

    The K problems never couple, so summing their losses and calling backward() once
    gives each realization exactly the gradient it would get on its own.

    Note: Learning rate scheduler implemented (not in gp.py). 

    Inputs:
        raw_params: (K, 3 + D); torch.nn.Parameter
        kernel: batched kernel, e.g. sq_exp_batch
        X_train: (K, n, D) -- one design matrix per realization
        base_params: (n, 3) [mu_th, mB, y_noise_meas] -- shared across realizations
    Outputs:
        list of K dicts, same format as optimize_GP
    """
    def _log_marginal_likelihood(raw_params, kernel, X_train, base_params):
        K_, n, D = X_train.shape
        mu_th, mB, y_noise_meas = base_params.T

        # Extract parameters (column per realization)
        M0 = raw_params[:, 0]
        sigma_int = torch.exp(raw_params[:, 1])
        sigma_f = torch.exp(raw_params[:, 2])
        ell = torch.exp(raw_params[:, 3:3 + D])

        # y = y(M0); each realization has its own M0
        y = (mu_th - (mB - M0[:, None])).unsqueeze(-1).to(dtype)        # (K, n, 1)

        y_noise_tot = (y_noise_meas + (sigma_int ** 2)[:, None]).to(dtype)   # (K, n)

        # Kernel
        params = {"sigma_f": sigma_f, "ell": ell}
        K = kernel(X_train, X_train, params)
        Ky = K + torch.diag_embed(y_noise_tot)                           # diag_embed replaces torch.diag

        # Cholesky -- factorizes each block independently
        L = torch.linalg.cholesky(Ky, upper=False)                       # (K, n, n)

        # Solve for alpha
        b = torch.linalg.solve_triangular(L, y, upper=False)
        alpha = torch.linalg.solve_triangular(L.transpose(-1, -2), b, upper=True)

        # Log marginal likelihood, one per realization
        t1 = -0.5 * (y * alpha).sum((-2, -1))                            # replaces y.T @ alpha
        t2 = -torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        t3 = -0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=dtype))
        return t1 + t2 + t3                                              # (K,)

    optimizer = torch.optim.Adam([raw_params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for step in range(epochs):
        optimizer.zero_grad()
        p = _log_marginal_likelihood(raw_params, kernel, X_train, base_params)
        loss = -p.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            print(step, -p.detach().mean().item())

    return [{'M0': raw_params[k, 0].item(),
             'sigma_int': torch.exp(raw_params[k, 1]).item(),
             'sigma_f': torch.exp(raw_params[k, 2]).item(),
             'ell': torch.exp(raw_params[k, 3:]).detach().cpu().numpy()}
            for k in range(raw_params.shape[0])]

def safe_cholesky_batch(K, jitter=1e-8, max_tries=5, upper=False):
    """Batched safe_cholesky. Adds jitter until EVERY matrix in the batch factorizes.
    Inputs:
        K: covariance matrices; shape (B, n, n)
        jitter: starting jitter; default 1e-8
        max_tries: maximum attempts to scale jitter by
        upper: L upper or lower triangular; default False
    Outputs:
        L: Cholesky decompositions, lower triangular; shape (B, n, n)
    """
    I = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)

    for i in range(max_tries):
        jitter_i = jitter * (10 ** i)
        print('jitter:', jitter_i)
        try:
            return torch.linalg.cholesky(K + jitter_i * I, upper=upper)
        except torch.linalg.LinAlgError:
            pass

    print('jitter / variance:', jitter_i / torch.diagonal(K, dim1=-2, dim2=-1).median())

    evals = torch.linalg.eigvalsh(K)
    raise RuntimeError(
        f"Cholesky failed. min eigenvalue={evals.min().item():.3e}, "
        f"jitter tried up to {jitter_i:.3e}"
    )

def predict_batch(kernel, X_train, y_train, X_test, params,
                  y_train_noise=None, jitter=1e-13, max_tries=4, diag_only=False):
    """Batched predict: K independent GPs at once.
    Inputs:
        kernel: batched kernel, e.g. sq_exp_batch
        X_train: (K, n, D)
        y_train: (K, n, 1)
        X_test: (K, m, D)
        params:
            - sigma_f: (K,)
            - ell: (K, D)
        y_train_noise: sigma_n^2; (n,) shared across realizations, or (K, n). should be y_train_e (og variance = mB**2 + redshift_err**2) + sigma_int**2.  
        diag_only: return only the predictive variance (K, m) instead of the full
                   covariance (K, m, m). Assumes k(x,x) = sigma_f**2, true for sq_exp.
    Outputs:
        f_pred: (K, m, 1)
        cov_pred: (K, m, m), or (K, m) if diag_only
        p: log marginal likelihood; (K,)
    """
    # Helper variables
    K_, n, D = X_train.shape
    m = X_test.shape[1]

    # Reshape y_noise to diagonal matrices. Compute training covariance.
    if y_train_noise is not None:
        noise = y_train_noise.expand(K_, n) if y_train_noise.dim() == 1 else y_train_noise
        Kmat = kernel(X_train, X_train, params) + torch.diag_embed(noise)
    else:
        Kmat = kernel(X_train, X_train, params) + jitter * torch.eye(
            n, device=X_train.device, dtype=X_train.dtype)

    # Cross-covariance block
    Ks = kernel(X_train, X_test, params)                              # (K, n, m)

    # Cholesky
    L = safe_cholesky_batch(Kmat, jitter=jitter, max_tries=max_tries, upper=False)

    # Solve for alpha
    b = torch.linalg.solve_triangular(L, y_train, upper=False)
    alpha = torch.linalg.solve_triangular(L.transpose(-1, -2), b, upper=True)

    # Predictive mean
    f_pred = Ks.transpose(-1, -2) @ alpha                             # (K, m, 1)

    # Predictive (co)variance
    v = torch.linalg.solve_triangular(L, Ks, upper=False)             # (K, n, m)
    if diag_only:
        kss_diag = (params["sigma_f"] ** 2)[:, None].expand(K_, m)
        cov_pred = kss_diag - (v ** 2).sum(-2)                        # (K, m)
    else:
        Kss = kernel(X_test, X_test, params)                          # (K, m, m)
        cov_pred = Kss - v.transpose(-1, -2) @ v                      # (K, m, m)

    # Log marginal likelihood
    t1 = -0.5 * (y_train * alpha).sum((-2, -1))
    t2 = -torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    t3 = -0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=Kmat.dtype))
    p = t1 + t2 + t3                                                  # (K,)

    return f_pred, cov_pred, p