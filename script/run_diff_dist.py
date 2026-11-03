"""Train one diffusion model per host-parameter realization, then sample.

For each realization i:
  1. build X from realization i of the host chains
  2. train a score model on it from scratch
  3. draw N samples per SN from that model
  4. save

merge() pools all K*N draws per SN and takes the median.

Chunked and checkpointed, so the run resumes after a wall-clock kill.

NOTE on cost: diffusion.train() loops over four (batch_size, lr) stages and
runs `epochs` epochs inside each, so --epochs 200 is really 800 passes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data import load_hierarchical_gp_data
from diffusion import __init__ as init_score_model, train, predict_SN, VESDE
from run_gp_dist import atomic_save, summarize, load_tripp_thresholds


def design_matrix(data, i, dtype):
    """Model inputs for realization i; shape (n, 7).

    Columns: x1, c, log10M, metal, log10sSFR, dust2, y_base.
    The first 6 are the conditionals (n_cond=6); y_base is the single target.
    """
    X_unsc = torch.vstack((
        data["x1_cat"],
        data["c_cat"],
        data["log10M_remain_thin"][:, i],
        data["mcmc_metal_thin"][:, i],
        data["log10sSFR_thin"][:, i],
        data["mcmc_dust2_thin"][:, i],
    )).T
    y_base = (data["mu_cat"] - data["mB_cat"]).reshape(-1, 1)
    return torch.hstack((X_unsc, y_base)).to(dtype)


def fit_and_sample(data, i, args, dtype):
    """Train a model on realization i from scratch and draw N samples per SN.

    Returns (draws (n, N), M0, final_loss).
    """
    X = design_matrix(data, i, dtype)

    # Same seed for every realization: initialization and shuffling are then
    # common-mode across realizations, so training stochasticity largely
    # cancels in the between-realization variance instead of inflating it.
    torch.manual_seed(args.seed)

    score_model, n_cond, n_dims = init_score_model(X, sde=VESDE(),
                                                   n_cond=args.n_cond,
                                                   device=args.device)
    assert n_dims == 1, f"expected a 1-d target, got n_dims={n_dims}"

    M_0_init = torch.tensor(-19.36, dtype=dtype, device=args.device)
    score_model, M0, losses, _ = train(
        score_model, X, n_cond,
        ratio_t=1.0, ratio_v=0.0, optimize_M0=True, M_0=M_0_init,
        plot=False, epochs=args.epochs,
    )

    y_pred, _, _ = predict_SN(score_model, X, n_cond,
                              n_samples=args.n_samples, M0=M0,
                              steps=args.steps, batch_size=args.batch_size)

    final_loss = float(losses[-1]) if len(losses) else float("nan")
    return y_pred.detach().cpu(), float(M0), final_loss


def merge(ckpt, out, y_base, hosts=None, thresholds=None, stat="mean"):
    """Pool all draws across realizations and collapse to a single prediction."""
    recs = [torch.load(c, map_location="cpu", weights_only=True)
            for c in sorted(ckpt.glob("chunk_*.pt"))]
    if not recs:
        return
    order = torch.argsort(torch.cat([r["idx"] for r in recs]))

    draws = torch.cat([r["draws"] for r in recs])[order]        # (K, n, N)
    M0_all = torch.cat([r["M0"] for r in recs])[order]          # (K,)
    loss_all = torch.cat([r["loss"] for r in recs])[order]      # (K,)
    K, n, N = draws.shape

    # per-realization prediction: median over that realization's N draws
    y_per_real = draws.median(dim=2).values                     # (K, n)

    # marginalized prediction: median over ALL K*N pooled draws
    y_med = draws.permute(1, 0, 2).reshape(n, K * N).median(dim=1).values

    resid_all = y_per_real - y_base[None, :]                    # (K, n)
    resid_med = y_med - y_base                                  # (n,)
    resid_bar = y_per_real.mean(0) - y_base                     # (n,)

    print(f"\nK={K}  N={N}  pooled draws per SN: {K * N}")
    print(f"M0    median {M0_all.median():.4f}  std {M0_all.std():.4f}")
    print(f"loss  median {loss_all.median():.5f}  std {loss_all.std():.5f}", flush=True)

    summary = summarize(resid_bar, resid_med, resid_all,
                        hosts=hosts, thresholds=thresholds, stat=stat)

    atomic_save({"draws": draws, "y_per_real": y_per_real, "y_med": y_med,
                 "resid": resid_all, "resid_med": resid_med, "resid_bar": resid_bar,
                 "M0": M0_all, "loss": loss_all,
                 "y_base": y_base, "summary": summary},
                out / "SDE4_dist_predictions.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=None)       # default: all realizations
    ap.add_argument("--chunk", type=int, default=16,
                    help="realizations per checkpoint file")
    ap.add_argument("--n-samples", type=int, default=1000)  # N draws per SN per realization
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-cond", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tripp-dir", type=Path, default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = args.out / "chunks"
    ckpt.mkdir(parents=True, exist_ok=True)

    data = load_hierarchical_gp_data(dtype=dtype, device=args.device)
    n_real = data["log10M_remain_thin"].shape[1]
    K = n_real if args.K is None else min(args.K, n_real)
    print(f"{n_real} realizations available, training {K} models", flush=True)

    for start in tqdm(list(range(0, K, args.chunk)), desc="chunks", unit="chunk"):
        idx = list(range(start, min(start + args.chunk, K)))
        path = ckpt / f"chunk_{start:05d}.pt"
        if path.exists():
            continue

        draws, M0s, losses = [], [], []
        for i in idx:
            d, m0, loss = fit_and_sample(data, i, args, dtype)
            draws.append(d)
            M0s.append(m0)
            losses.append(loss)

        atomic_save({"idx": torch.tensor(idx),
                     "draws": torch.stack(draws),               # (chunk, n, N)
                     "M0": torch.tensor(M0s),
                     "loss": torch.tensor(losses)}, path)

        tqdm.write(f"chunk {start}: M0 median {torch.tensor(M0s).median():.4f}  "
                   f"loss median {torch.tensor(losses).median():.5f}")

    y_base = (data["mu_cat"] - data["mB_cat"]).flatten().cpu()
    merge(ckpt, args.out, y_base,
          hosts=(data["log10M_remain_med"].cpu(), data["log10sSFR_med"].cpu()),
          thresholds=load_tripp_thresholds(args.tripp_dir) if args.tripp_dir else None)


if __name__ == "__main__":
    main()
