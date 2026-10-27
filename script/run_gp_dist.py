"""Fit K GP realizations over the thinned host-parameter posteriors.

For each realization i:
  1. build the design matrix X from realization i of the host chains
  2. fit [M0, sigma_int, sigma_f, ell] by maximum marginal likelihood
  3. predict the correction at every SN

Noise enters in two places and they are deliberately different:
  - TRAINING covariance uses y_noise_e + sigma_int**2, matching the
    likelihood that was maximized. Conditioning on anything else would
    mean predicting from a different model than the one that was fitted.
  - TEST points get no noise added. predict_batch returns Var[f*], the
    variance of the latent correction itself, which is what we want.
    Adding sigma_int**2 there would give Var[y*] for a hypothetical new
    SN -- a larger and different quantity.

Results are checkpointed per chunk, so the run is resumable.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from data import load_hierarchical_gp_data
from kernels import sq_exp_batch
from batched_gp import optimize_GP_batch, predict_batch, stack_params, to_raw, from_raw


D = 6      # number of GP input features; must match design_matrix


def atomic_save(obj, path):
    """Write via temp file + rename so a crash cannot truncate an existing file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def design_matrix(data, i, dtype):
    """GP inputs for realization i; shape (n, 6).

    Column order must stay [x1, c, log10M, metal, log10sSFR, dust2].
    """
    return torch.vstack((
        data["x1_cat"],
        data["c_cat"],
        data["log10M_remain_thin"][:, i],
        data["mcmc_metal_thin"][:, i],
        data["log10sSFR_thin"][:, i],
        data["mcmc_dust2_thin"][:, i],
    )).T.to(dtype)


def cold_init_vec(data, dtype, device):
    """Neutral start: M0 = -19.36, sigma_int = y_std, sigma_f = 1, ell = 1 (all features)."""
    return torch.tensor(
        [-19.36, float(np.log(float(data["y_std"]))), 0.0] + [0.0] * D,
        dtype=dtype, device=device,
    )


def fit_chunk(Xb, init_vec, base_params, args):
    """Fit one batch of realizations. Returns (list of param dicts, raw tensor)."""
    raw = torch.nn.Parameter(init_vec.repeat(Xb.shape[0], 1).clone())
    params_list = optimize_GP_batch(
        raw, sq_exp_batch, Xb, base_params,
        epochs=args.epochs, lr=args.lr, gamma=args.gamma,
    )
    return params_list, raw


def predict_chunk(Xb, params_list, data, dtype, device, diag_only):
    """Predict the correction at every SN, for every realization in the chunk.

    Returns (mu, cov, lml) with mu shape (chunk, n).
    """
    P, M0 = stack_params(params_list, dtype=dtype, device=device)
    sigma_int = torch.tensor([p["sigma_int"] for p in params_list],
                             dtype=dtype, device=device)

    # target, rebuilt with each realization's own fitted M0
    y = (data["mu_cat"] - (data["mB_cat"] - M0[:, None])).unsqueeze(-1).to(dtype)

    # training noise: measurement variance + fitted intrinsic scatter.
    # y_noise_e is already a VARIANCE, so sigma_int enters squared, not in quadrature.
    y_noise_tot = data["y_noise_e"][None, :] + (sigma_int ** 2)[:, None]   # (chunk, n)

    f_pred, cov_pred, lml = predict_batch(
        sq_exp_batch, Xb, y, Xb, P,
        y_train_noise=y_noise_tot, jitter=1e-13, diag_only=diag_only,
    )

    mu = f_pred.squeeze(-1) - M0[:, None]      # the correction, with M0 removed

    return mu, cov_pred, lml


def load_tripp_thresholds(tripp_dir):
    """Step positions from the Tripp MCMC fits."""
    tripp_dir = Path(tripp_dir)
    mass   = torch.load(tripp_dir / "Tripp_mass_params.pt",    weights_only=False)
    massab = torch.load(tripp_dir / "Tripp_mass_ab_params.pt", weights_only=False)
    ssfr   = torch.load(tripp_dir / "Tripp_sSFR_params.pt",    weights_only=False)

    t_mass   = float(mass[-1])
    t_massab = float(massab["tracer_thresh"]) if isinstance(massab, dict) else float(massab[-1])
    t_ssfr   = float(ssfr[-1])
    return t_mass, t_massab, t_ssfr


def step_bias(residual, x_host, thresh, stat="mean"):
    """stat(residual | host >= thresh) - stat(residual | host < thresh)."""
    low, high = x_host < thresh, x_host >= thresh
    if not low.any() or not high.any():
        return float("nan")
    f = torch.mean if stat == "mean" else torch.median
    return float(f(residual[high]) - f(residual[low]))


def summarize(resid_bar, resid_med, resid_all, hosts=None, thresholds=None, stat="mean"):
    """Summary statistics of the residuals, computed three ways.

    mean-collapse   : statistic of resid_bar  (realization-MEAN residual)
    median-collapse : statistic of resid_med  (realization-MEDIAN residual)
    per-realization : statistic computed separately for each realization, then
                      summarized as median +- std. That std is the host-parameter
                      uncertainty on the statistic -- the number this whole
                      marginalization exists to measure.

    The collapsed columns are always <= the per-realization median for RMS,
    because averaging cancels realization-specific noise. Quote both.
    """
    K = resid_all.shape[0]
    rows = []

    # --- scalar statistics, vectorized over the realization axis ---
    for name, f_one, f_many in [
        ("median", lambda r: r.median(),             lambda R: R.median(dim=1).values),
        ("mean",   lambda r: r.mean(),               lambda R: R.mean(dim=1)),
        ("rms",    lambda r: r.pow(2).mean().sqrt(), lambda R: R.pow(2).mean(dim=1).sqrt()),
        ("std",    lambda r: r.std(),                lambda R: R.std(dim=1)),
    ]:
        rows.append((name, f_one(resid_bar), f_one(resid_med), f_many(resid_all)))

    # --- host step biases: stat(high) - stat(low) at each Tripp threshold ---
    if hosts is not None and thresholds is not None:
        log10M, log10sSFR = hosts
        t_mass, t_massab, t_ssfr = thresholds

        for label, host, thresh in [
            ("bias mass @ Tripp+mass",    log10M,    t_mass),
            ("bias mass @ Tripp+mass+ab", log10M,    t_massab),
            ("bias sSFR @ Tripp+sSFR",    log10sSFR, t_ssfr),
        ]:
            low, high = host < thresh, host >= thresh
            if stat == "mean":
                one = lambda r: r[high].mean() - r[low].mean()
                many = resid_all[:, high].mean(1) - resid_all[:, low].mean(1)
            else:
                one = lambda r: r[high].median() - r[low].median()
                many = (resid_all[:, high].median(1).values
                        - resid_all[:, low].median(1).values)
            rows.append((f"{label} ({thresh:.2f})", one(resid_bar), one(resid_med), many))

    # --- print ---
    print(f"\n{'statistic':<34} {'mean-col':>10} {'median-col':>11} "
          f"{'per-real':>10} {'host sig':>10}")
    print("-" * 78)
    for name, v_bar, v_med, per_k in rows:
        print(f"{name:<34} {v_bar:>10.5f} {v_med:>11.5f} "
              f"{per_k.median():>10.5f} {per_k.std():>10.5f}", flush=True)

    return {name: {"mean_collapse": float(v_bar),
                   "median_collapse": float(v_med),
                   "per_real_median": float(per_k.median()),
                   "host_sigma": float(per_k.std())}
            for name, v_bar, v_med, per_k in rows}

def merge(ckpt, out, y_base, device="cpu", hosts=None, thresholds=None, stat="mean"):
    """Combine chunk files and marginalize over realizations.

    device: where to do the arithmetic. Results are moved back to CPU before
    saving so the output files load anywhere.
    """
    recs = [torch.load(c, map_location=device, weights_only=True)
            for c in sorted(ckpt.glob("chunk_*.pt"))]
    if not recs:
        return
    order = torch.argsort(torch.cat([r["idx"] for r in recs]))

    raw_all = torch.cat([r["raw"] for r in recs])[order]
    mu_all  = torch.cat([r["mu"]  for r in recs])[order]
    cov_all = torch.cat([r["cov"] for r in recs])[order]
    lml_all = torch.cat([r["lml"] for r in recs])[order]
    y_base  = y_base.to(mu_all.device)

    K = mu_all.shape[0]
    var_within = (torch.diagonal(cov_all, dim1=-2, dim2=-1)
                  if cov_all.dim() == 3 else cov_all).clamp_min(0)

    # --- the correction, collapsed over realizations two ways ---
    mu_bar = mu_all.mean(0)                      # (n,)
    mu_med = mu_all.median(dim=0).values         # (n,)

    # --- residual = correction - (mu - mB), matching the notebook convention ---
    resid_all = mu_all - y_base[None, :]         # (K, n)
    resid_bar = mu_bar - y_base                  # (n,)
    resid_med = mu_med - y_base                  # (n,)

    # --- law of total variance: GP uncertainty + host-parameter uncertainty ---
    B = mu_all.var(0, correction=1)
    var_tot = var_within.mean(0) + (1 + 1 / K) * B
    lam = (1 + 1 / K) * B / var_tot

    print(f"\nK={K}   NLL median {(-lml_all).median():.4f}  std {(-lml_all).std():.4f}")
    print(f"lambda (host-param fraction of variance)  "
          f"median {lam.median():.4f}  max {lam.max():.4f}", flush=True)

    summary = summarize(resid_bar.cpu(), resid_med.cpu(), resid_all.cpu(),
                        hosts=None if hosts is None else tuple(h.cpu() for h in hosts),
                        thresholds=thresholds, stat=stat)

    atomic_save({"raw": raw_all.cpu(),
                 "params": [from_raw(raw_all[k].cpu()) for k in range(K)],
                 "lml": lml_all.cpu()},
                out / "GP4_dist_optimal_params.pth")
    atomic_save({"mu": mu_all.cpu(), "cov": cov_all.cpu(),
                 "mu_bar": mu_bar.cpu(), "mu_med": mu_med.cpu(),
                 "resid": resid_all.cpu(),
                 "resid_bar": resid_bar.cpu(),
                 "resid_med": resid_med.cpu(),
                 "y_base": y_base.cpu(),
                 "var_tot": var_tot.cpu(), "lam": lam.cpu(),
                 "summary": summary},
                out / "GP4_dist_predictions.pth")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    type=Path, required=True)
    ap.add_argument("--init",   type=Path, default=None,
                    help="warm start from a saved param dict; omit for cold start")
    ap.add_argument("--K",      type=int, default=None)     # default: all realizations
    ap.add_argument("--chunk",  type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr",     type=float, default=0.05)
    ap.add_argument("--gamma",  type=float, default=0.998)
    ap.add_argument("--dtype",  default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge-device", default=None,
                    help="device for the merge step; defaults to --device")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--tripp-dir", type=Path, default=None,
                help="directory holding Tripp_*_params.pt; omit to skip bias report")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    ckpt = args.out / "chunks"
    ckpt.mkdir(parents=True, exist_ok=True)

    data = load_hierarchical_gp_data(dtype=dtype, device=args.device)

    n_real = data["log10M_remain_thin"].shape[1]
    K = n_real if args.K is None else min(args.K, n_real)
    print(f"{n_real} realizations available, fitting {K}", flush=True)

    base_params = torch.vstack((data["mu_cat"], data["mB_cat"], data["y_noise_e"])).T

    if args.init is None:
        init_vec = cold_init_vec(data, dtype, args.device)
        print("cold start:", [round(v, 4) for v in init_vec.tolist()], flush=True)
    else:
        init_vec = to_raw(torch.load(args.init, weights_only=False),
                          dtype=dtype, device=args.device)
        print(f"warm start from {args.init}", flush=True)

    for start in range(0, K, args.chunk):
        idx = list(range(start, min(start + args.chunk, K)))
        path = ckpt / f"chunk_{start:05d}.pt"
        if path.exists():
            print(f"chunk {start}: done, skipping", flush=True)
            continue
        print(f"\n=== chunk {start}: realizations {idx[0]}-{idx[-1]} ===", flush=True)

        Xb = torch.stack([design_matrix(data, i, dtype) for i in idx])

        params_list, raw = fit_chunk(Xb, init_vec, base_params, args)
        mu, cov, lml = predict_chunk(Xb, params_list, data, dtype,
                                     args.device, args.diag_only)

        atomic_save({"idx": torch.tensor(idx),
                     "raw": raw.detach().cpu(),
                     "mu":  mu.detach().cpu(),
                     "cov": cov.detach().cpu(),
                     "lml": lml.detach().cpu()}, path)

    y_base = (data["mu_cat"] - data["mB_cat"]).flatten().cpu()
    merge(ckpt, args.out, y_base,
          device=args.merge_device or args.device,
          hosts=(data["log10M_remain_med"], data["log10sSFR_med"]),
          thresholds=load_tripp_thresholds(args.tripp_dir) if args.tripp_dir else None)


if __name__ == "__main__":
    main()