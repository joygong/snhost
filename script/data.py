from pathlib import Path

import numpy as np
import torch


DEFAULT_ZTF_DIR = Path("/home/jjg65/rds/hpc-work/ztf_sn")
DEFAULT_GAL_DIR = Path("/home/jjg65/rds/hpc-work/ztf_meds")


def _to_tensor(x, *, dtype=torch.float64, device=None):
    return torch.as_tensor(x, dtype=dtype, device=device)

def load_npz(path, keys, *, dtype=torch.float64, device=None):
    """Load selected keys from a .npz file as torch tensors."""
    path = Path(path)
    with np.load(path) as data:
        return {
            key: _to_tensor(data[key], dtype=dtype, device=device)
            for key in keys
        }

def load_npy(path, *, dtype=torch.float64, device=None):
    """Load a .npy file as a torch tensor."""
    return _to_tensor(np.load(Path(path)), dtype=dtype, device=device)


def load_hierarchical_gp_data(
    ztf_dir=DEFAULT_ZTF_DIR,
    gal_dir=DEFAULT_GAL_DIR,
    *,
    dtype=torch.float64,
    device=None,
):
    """Load ZTF SN data and host medians for the hierarchical GP model."""
    ztf_dir = Path(ztf_dir)
    gal_dir = Path(gal_dir)

    data = {}
    data.update(load_npz(
        ztf_dir / "z_param.npz",
        ["z_cmb_cat", "zcmb_cat_err"],
        dtype=dtype,
        device=device,
    ))
    data.update(load_npz(
        ztf_dir / "sn_param.npz",
        ["mB_cat", "x1_cat", "c_cat"],
        dtype=dtype,
        device=device,
    ))
    data.update(load_npz(
        ztf_dir / "sn_param_err.npz",
        ["sig_mB", "sig_x1", "sig_c", "sig_mBx1", "sig_mBc", "sig_x1c"],
        dtype=dtype,
        device=device,
    ))
    data.update(load_npz(
        ztf_dir / "mu_param.npz",
        ["mu_cat"],
        dtype=dtype,
        device=device,
    ))

    data.update(load_npz(
        ztf_dir / "gp_param.npz",
        ["y_std", "y_noise_e"],
        dtype=dtype,
        device=device,
    ))

    for name in [
        "log10M_remain_med",
        "mcmc_metal_med",
        "log10sSFR_med",
        "mcmc_dust2_med",
    ]:
        data[name] = load_npy(gal_dir / f"{name}.npy", dtype=dtype, device=device)

    data["thetas"] = torch.column_stack([
        data["log10M_remain_med"],
        data["mcmc_metal_med"],
        data["log10sSFR_med"],
        data["mcmc_dust2_med"],
    ])

    # load thinned chains: already in pt format  
    chains = torch.load(gal_dir / "thin_chains_patched.pt",
                        map_location="cpu", weights_only=True)
    for name in [
        "log10M_remain",
        "mcmc_metal",
        "log10sSFR",
        "mcmc_dust2",
    ]:
        data[f"{name}_thin"] = chains[name].to(dtype=dtype, device=device)
    
    data["yhat"] = torch.cat([
        data["mB_cat"],
        data["x1_cat"],
        data["c_cat"],
    ])

    return data
