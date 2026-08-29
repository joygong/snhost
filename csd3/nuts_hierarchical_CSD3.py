import argparse

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 

from data import load_hierarchical_gp_data
from kernels import sq_exp

dtype = None
device = None

def scalar(x):
    return torch.tensor(x, dtype=dtype, device=device)

def zeros(*shape):
    return torch.zeros(*shape, dtype=dtype, device=device)

def ones(*shape):
    return torch.ones(*shape, dtype=dtype, device=device)

def eye(n):
    return torch.eye(n, dtype=dtype, device=device)

def calculate_sig_mu_th(z, sig_z, sig_pec=200):        
    """Distance-modulus uncertainty from redshift and peculiar velocity.
    """
    c = (const.c).to(u.km / u.s).value
    return torch.sqrt((5 / (z * torch.log(scalar(10.))))**2 * ((sig_pec / c)**2 + sig_z**2))   

def permute_cov(W, n_sn, d=3):
    """Reorder block-diagonal per-SN covariances to [all mB, all x1, all c].
    """
    idx = torch.arange(n_sn * d, device=W.device).reshape(n_sn, d).T.reshape(-1)
    return W[idx][:, idx]

def build_K(H, thetas):
    """Build prior covariance matrices K_gamma, K_psi, K_phi given GP hyperparameters. 
    """
    K_gamma = sq_exp(thetas, thetas, params=H["gamma"])   
    K_psi = sq_exp(thetas, thetas, params=H["psi"])
    K_phi = sq_exp(thetas, thetas, params=H["phi"])

    K = {"K_gamma": K_gamma, 
         "K_psi": K_psi,
         "K_phi": K_phi}

    return K

def build_C(H, thetas, s, WM, force_sym=True): 
    """Build final covariance matrix C from hyperparameters H, host parameters theta, mu uncertainties s, and SALT2-permuted covariances WM.
    Parameters: 
        H: hyperparameters (21) [most efficiently without dust hyperparameters a & b]; dict
        thetas: host parameters; (n_sn, 4)
        s: diagonal matrix of distance modulus error assumed to come only from z and pec_z; (n_sn,)
        WM: permuted SALT2 covariance matrix; (3*n, 3*n)

    Returns: 
        C: covariance matrix; (3*n, 3*n)
    """  
    # Helpers
    n_sn = thetas.shape[0]
    I = eye(n_sn)

    # K  
    K = build_K(H, thetas)
    K_gamma, K_psi, K_phi = K["K_gamma"], K["K_psi"], K["K_phi"]

    # A, B 
    A = torch.vstack((H["alpha_x"] * I, H["beta_c"] * I)).T
    B_1 = torch.hstack((K_phi + H["sigma_x"]**2 * I, H["alpha_c"] * (K_phi + H["sigma_x"]**2 * I)),)
    B_2 = torch.hstack((H["alpha_c"] * (K_phi + H["sigma_x"]**2 * I), H["alpha_c"]**2 * (K_phi + H["sigma_x"]**2 * I) + K_psi + H["sigma_c"]**2 * I),)
    B = torch.vstack((B_1, B_2))

    # S
    S11 = A @ B @ A.T + K_gamma + I * H["sigma_0"]**2 + s
    S12 = A @ B
    S21 = B @ A.T
    S22 = B 
    S = torch.vstack((
        torch.hstack((S11, S12)),
        torch.hstack((S21, S22)),
    ))

    C = S + WM
    if force_sym:                # to account for tiny floating-point errors
        C = 0.5 * (C + C.T)

    return C

def model(yhat, thetas, mu_lcdm, sig_mu, WM, jitter=1e-8, tau_const=True, dust=True, model_type="marginalized"):
    """Define hierarchical model and likelihood for inference. 
    Parameters: 
        yhat: observed SN parameters, flattened; (3*n_sn,)
        thetas: host parameters; (n_sn, 4)
        mu_lcdm: cosmologically predicted distance modulus; (n_sn,) 
        sig_mu: distance modulus uncertainty assumed to come only from z and pec_z; (n_sn,). [NOTE: non-diagonalized, not a matrix!]
        WM: permuted SALT2 covariance matrix; (3*n, 3*n)
        jitter: minute numerical value added for stability; default 1e-8
        model_type: 'marginalized' or 'explicit'
            - 'marginalized': likelihood = p(yhat | E_BV, H) with covariance S + WM
            - 'explicit': likelihood = p(yhat | latent true y) p(latent true y | H, E_BV) with covariance WM 

    Returns: None
    """
    n_sn = thetas.shape[0]
    I = eye(n_sn)

    # (1) Hyperpriors
        # Cannot truly be drawn from -inf, inf, so use big number approximations  
    # SN/global hyperparameters
    alpha_x = pyro.sample("alpha_x", dist.Uniform(scalar(-1.), scalar(1.)))
    alpha_c = pyro.sample("alpha_c", dist.Uniform(scalar(-1.), scalar(1.)))
    beta_c  = pyro.sample("beta_c",  dist.Uniform(scalar(-5.), scalar(5.)))

    # sigma_0 = pyro.sample("sigma_0", dist.Uniform(scalar(0.), scalar(100.)))
    sigma_0 = pyro.sample("sigma_0", dist.HalfNormal(scalar(2.0)))
    # sigma_x = pyro.sample("sigma_x", dist.Uniform(scalar(0.), scalar(100.)))
    sigma_x = pyro.sample("sigma_x", dist.HalfNormal(scalar(2.0)))
    # sigma_c = pyro.sample("sigma_c", dist.Uniform(scalar(0.), scalar(100.)))
    sigma_c = pyro.sample("sigma_c", dist.HalfNormal(scalar(2.0)))

    M_0 = pyro.sample("M_0", dist.Uniform(scalar(-21.), scalar(-17.)))
    R_B = pyro.sample("R_B", dist.Uniform(scalar(1.), scalar(5.))) if dust else scalar(0.)

    # GP kernel hyperparameters: for D=4 host inputs, each has 1 sigma_f + 4 ell
    gamma_sf  = pyro.sample("gamma_sf",  dist.Uniform(scalar(0.), scalar(100.)))
    gamma_ell = pyro.sample("gamma_ell", dist.Uniform(zeros(4), ones(4) * 100.).to_event(1))

    psi_sf  = pyro.sample("psi_sf",  dist.Uniform(scalar(0.), scalar(100.)))
    psi_ell = pyro.sample("psi_ell", dist.Uniform(zeros(4), ones(4) * 100.).to_event(1))

    phi_sf  = pyro.sample("phi_sf",  dist.Uniform(scalar(0.), scalar(100.)))
    phi_ell = pyro.sample("phi_ell", dist.Uniform(zeros(4), ones(4) * 100.).to_event(1))

    # Build hyperparameters H_nodust to feed into covariance C
    H_nodust = {
        "alpha_x": alpha_x,
        "alpha_c": alpha_c,
        "beta_c": beta_c,
        "sigma_0": sigma_0,
        "sigma_x": sigma_x,
        "sigma_c": sigma_c,
        "gamma": {"sigma_f": gamma_sf, "ell": gamma_ell},
        "psi": {"sigma_f": psi_sf, "ell": psi_ell},
        "phi": {"sigma_f": phi_sf, "ell": phi_ell},
    }

    # Define plate for use
    sn_plate = pyro.plate("sn", n_sn)

    # Dust population hyperparameters
    if dust: 
        if tau_const:
            # tau = pyro.sample("tau", dist.Uniform(scalar(0.), scalar(100.)))
            tau = pyro.sample("tau", dist.HalfNormal(scalar(1.0)))
            tau_e = tau.expand(n_sn)
        else:
            a = pyro.sample("a", dist.Uniform(-scalar(100.), scalar(100.)))
            b = pyro.sample("b", dist.Uniform(-scalar(100.), scalar(100.)))
            tau_2 = thetas[:, -1]
            tau_e = torch.nn.functional.softplus(a * tau_2 + b)         # + jitter

        pyro.deterministic("tau_e", tau_e)              # save one value per SN 

        # (2) Latent dust for every SN
        with sn_plate:
            E_BV = pyro.sample("E_BV", dist.Exponential(rate=ones(n_sn) / tau_e)) 
    
    else: 
        E_BV = zeros(n_sn)

    # (3) Analytically marginalized version: likelihood = p(yhat | H, E_BV) 
    if model_type == "marginalized":
        s = torch.diag(sig_mu**2)

        nu = torch.cat([
            M_0 * ones(n_sn) + mu_lcdm + R_B * E_BV,
            zeros(n_sn),
            E_BV,
        ])                                      # (1899,)

        C = build_C(H_nodust, thetas, s, WM)
        L = torch.linalg.cholesky(C + jitter * eye(C.shape[0]), upper=False)    # add jitter if needed 

        pyro.sample(
            "yhat",
            dist.MultivariateNormal(loc=nu, scale_tril=L),      # can pass one of: covariance_matrix, precision_matrix, or scale_tril [most efficient; lower-triangular factor of covariance w/ positive-valued diagonal]
            obs=yhat,   
        )
    
    # (3) Explicit latent prediction version: likelihood = p(yhat | latent true y) p(latent true y | H, E_BV)
    elif model_type == "explicit":
        K = build_K(H_nodust, thetas)
        K_gamma, K_psi, K_phi = K["K_gamma"], K["K_psi"], K["K_phi"]

        x1 = pyro.sample("x1", dist.MultivariateNormal(zeros(n_sn), K_phi + sigma_x**2 * I))
        c_int_resid = pyro.sample("c_int_resid", dist.MultivariateNormal(zeros(n_sn), K_psi + sigma_c**2 * I))
        M_int_resid = pyro.sample("M_int_resid", dist.MultivariateNormal(zeros(n_sn), K_gamma + sigma_0**2 * I))
    
        # Latent mu for every SN 
        with sn_plate: 
            mu = pyro.sample("mu", dist.Normal(mu_lcdm.to(device), sig_mu.to(device)))   # note std is fed in, NOT variance 

        c_int = alpha_c * x1 + c_int_resid
        M_int = (M_0 + alpha_x * x1 + beta_c * c_int + M_int_resid)

        print("M_int device:", M_int.device)
        print("R_B device:", R_B.device)
        print("E_BV device:", E_BV.device)
        print("mu device:", mu.device)

        mB_true = M_int + R_B * E_BV + mu 
        x1_true = x1
        c_true = c_int + E_BV

        # Apparent (dust-extincted) latents
        pyro.deterministic("mB_true", mB_true)
        pyro.deterministic("x1_true", x1_true)
        pyro.deterministic("c_true", c_true)

        y_true = torch.cat([mB_true, x1_true, c_true])
        WM = 0.5 * (WM + WM.T)
        L_WM = torch.linalg.cholesky(WM + jitter * eye(WM.shape[0]), upper=False)   

        pyro.sample(
            "yhat",
            dist.MultivariateNormal(loc=y_true, scale_tril=L_WM),    # WM only; sig_mu is represented by the sampled latent mu
            obs=yhat,
        )

    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

def main(args): 
    global dtype, device

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if args.cuda else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    torch.set_default_dtype(dtype)
    pyro.set_rng_seed(args.rng_seed)

    # LOAD DATA
    data = load_hierarchical_gp_data(
        ztf_dir=args.ztf_dir,
        gal_dir=args.gal_dir,
        dtype=dtype,
        device=device,
    )

    # check to see if loaded in corrected covariances
    # print('sig_mB:', data["sig_mB"])

    thetas = data["thetas"]
    yhat = data["yhat"].reshape(-1)         # (3*n_sn,)
    n_sn = thetas.shape[0]
   
    sig_mu = calculate_sig_mu_th(data["z_cmb_cat"], data["zcmb_cat_err"])
    W = zeros(n_sn*3, n_sn*3)
    for i in range(n_sn): 
        l = 3*i
        h = 3*(i+1)
        cov_i = torch.stack([
            torch.stack([(data["sig_mB"]**2)[i], data["sig_mBx1"][i], data["sig_mBc"][i]]),
            torch.stack([data["sig_mBx1"][i], (data["sig_x1"]**2)[i], data["sig_x1c"][i]]),
            torch.stack([data["sig_mBc"][i], data["sig_x1c"][i], (data["sig_c"]**2)[i]]),
        ])
        W[l:h, l:h] = cov_i
    WM = permute_cov(W, n_sn=n_sn, d=3)

    cosmo = Planck18
    mu_lcdm = torch.as_tensor(
        cosmo.distmod(data["z_cmb_cat"].detach().cpu().numpy()).value,
        dtype=dtype,
        device=device,
    )

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"n_sn: {n_sn}")
    print("jitter =", args.jitter)
    print("tau_const =", args.tau_const)
    print("dust =", args.dust)
    print("model =", args.model)

    # RUN INFERENCE
    nuts_kernel = NUTS(
        model,    
        target_accept_prob=args.target_accept_prob,     # default 0.8 
        max_tree_depth=args.max_tree_depth,             # default 10 
    )

    mcmc = MCMC(
        nuts_kernel,
        num_samples=args.num_samples,              # 2000 -> will take a long time
        warmup_steps=args.warmup_steps,
        num_chains=args.num_chains,                # default: chains will run in parallel w/ jax.pmap()
    )

    mcmc.run(yhat, thetas, mu_lcdm, sig_mu, WM, jitter=args.jitter, tau_const=args.tau_const, dust=args.dust, model_type=args.model)

    # SAVE SAMPLES 
    torch.save(
        {
            "samples": mcmc.get_samples(group_by_chain=True),
            "summary": mcmc.summary(), 
            "diagnostics": mcmc.diagnostics(),
        },
        args.output,                                # f"nuts_{args.num_samples}.pt",       
    )

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Hierarchical Bayesian SN Ia with NUTS")

    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--warmup-steps", nargs="?", default=1000, type=int)
    parser.add_argument("--rng_seed", nargs="?", default=0, type=int)
    parser.add_argument("--target_accept_prob", nargs="?", default=0.85, type=float)
    parser.add_argument("--max_tree_depth", nargs="?", default=10, type=int)
    
    parser.add_argument("--ztf-dir", default="/home/jjg65/rds/hpc-work/ztf_sn/")
    parser.add_argument("--gal-dir", default="/home/jjg65/rds/hpc-work/ztf_meds/")
    # parser.add_argument("--ztf-dir", default="/Users/joygong/CAMBRIDGE/inputs/ztf/")
    # parser.add_argument("--gal-dir", default="/Users/joygong/CAMBRIDGE/outputs/gal_params/")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")                  
    parser.add_argument("--jitter", default=1e-8, type=float)
    parser.add_argument("--tau-const", action=argparse.BooleanOptionalAction, default=True, help="use one shared dust-population tau")
    parser.add_argument("--dust", action=argparse.BooleanOptionalAction, default=True, help="include separation of color into c_int and E_BV")
    parser.add_argument("--model", choices=["marginalized", "explicit"], default="marginalized")
    parser.add_argument("--output", default="/home/jjg65/rds/hpc-work/ztf_nuts/nuts_samples_explicit.pt")
    # parser.add_argument("--output", default="/Users/joygong/CAMBRIDGE/snhost/scratch/nuts_samples_explicit.pt")
    parser.add_argument("--cuda", action="store_true", help="run on GPU")

    args = parser.parse_args()

    main(args) 


    
