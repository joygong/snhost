from turtle import window_height
from pathlib import Path

import numpy as np
import torch 
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os
from snhost.scratch.shared import get_plot_metadata

_PLOT_METADATA = get_plot_metadata()
INPUT_INDICES = _PLOT_METADATA["input_indices"]
LATEX_LABELS = _PLOT_METADATA["latex_labels"]
_DEFAULT_TRIPP_PT_DIR = Path(__file__).resolve().parent.parent / "scratch"

MODEL_COLORS = {
    "Diffusion": "#882255",
    "VE SDE-4 OOS": "#66CCEE",
    "VE SDE 80": "#882255",
    # "VE SDE 100": "teal",
    "VE SDE 100": "#882255",
    "VE SDE OOS": "#66CCEE",
    "VE SDE-4 100": "#882255",
    "VE SDE-2 100": "#C7ABFF",
    "VE SDE-2 OOS": "#66CCEE",
    "VE SDE 100 2": "red",
    "VP SDE": "teal",
    "VP SDE ind": "teal",
    "Tripp+sSFR": "#117733",
    "Tripp+mass": "#117733",
    "Tripp+mass+ab": "#117733",
    "GP": "#DD8452",
    "GP-4 100": "#DD8452",
    # "GP-2 100": "#7D1313",
    "GP-2 100": "#FFF075",
    "GP 100": "#DD8452",
    "GP-4 OOS": "#66CCEE",
    "GP-2 OOS": "#66CCEE",
}

TRIPP_LINESTYLES = {
    "Tripp+sSFR": "-",
    "Tripp+mass": "--",
    "Tripp+mass+ab": ":",
    # "Tripp+mass+ab": "-.",
}

### Plot data and model predictions 
def plot1D_ax(ax, input_var, X_train, y_train, sub_M0=True, GP=True, GP_params=None, Tripp=True, Tripp_params=None, sde=True, sde_params=None, sde_cloud=False):
    """
    Plot a 1D marginal slice of data or model prediction as one input varies.
    Assumes all other inputs are fixed.
    Options to overplot GP, Tripp & diffusion model predictions. 
    Inputs: 
        ax: ax to plot 
        X_test: test points; shape (m, 6)
        input_var: input column to vary
        f_pred: predictive mean; shape (m,)
        cov_pred: predictive variance; shape (m,)
        p: log marginal likelihood; shape (m,)
        M0_GP: M0 optimized by GP; = M0_fit
        Tripp: 
            True: plot Tripp prediction 
        Tripp_params: 
            f_pred_tripp: ALL_tripp; dictionary with keys as inputs, each of shape (m,)
            M0_Tripp: M0 optimized by Tripp estimator; = M_B
        sde: 
            True: plot SDE prediction
        sde_params: 
            X_grids: conditional inputs for predictions; dictionary with keys as inputs, each of shape (600,)
            y_pred_sde: sde target predictions; dictionary with keys as inputs, each of shape (600, 1)
            M0_fit_sde: M0 optimized by SDE 
            x_medss: binned conditional input medians. 
            y_medss: binned conditional target medians. 
        sub_M0: 
            True: plot y = \mu - mB. 
            False: plot y = \mu - (mB - M0)
    Outputs: 
        1D plot 
    """
    i = INPUT_INDICES[input_var]

    x_train = X_train[:, i].detach().cpu()
    y_train = y_train.flatten().detach().cpu() 
    # if not sub_M0 else y_train.flatten().detach().cpu() - M0_GP 
    ax.scatter(x_train, y_train, s=18, alpha=0.08, color="black", label="_nolegend_")

    if GP == True: 
        M0_GP = GP_params['M0']
        f_pred_GP = GP_params['f_pred'].detach().cpu().flatten() if not sub_M0 else GP_params['f_pred'].detach().cpu().flatten() - M0_GP 
        X_grid_GP = GP_params['X_grid'].detach()
        x_grid_GP = X_test_GP[:, i]
        cov_pred = GP_params['cov_pred'].detach().cpu()
        var_pred = torch.diagonal(cov_pred)
        std_pred = torch.sqrt(torch.clamp(var_pred, min=0.0))

    # if sub_M0: 
    #     y_train = y_train - M0_GP
    #     f_pred = f_pred - M0_GP

        ax.plot(x_grid_GP, f_pred_GP, color="C1", lw=2, label="GP mean")
        ax.fill_between(
            x_test,
            f_pred - 2 * std_pred,
            f_pred + 2 * std_pred,
            color="C1",
            alpha=0.13,
            label="_nolegend_",
        )

    if Tripp == True and Tripp_params is not None: 
        F_pred_tripp = Tripp_params['f_pred']
        X_grid_tripp = Tripp_params['X_grid']
        M0_Tripp = Tripp_params['M0']
        f_pred_tripp = F_pred_tripp[input_var].detach().cpu() if not sub_M0 else F_pred_tripp[input_var].detach().cpu() - M0_Tripp

        ax.plot(X_grid_tripp, f_pred_tripp, color="C2", lw=2, label="Tripp+sSFR")

    if sde == True and sde_params is not None: 
        Y_pred_sde = sde_params['y_pred']
        X_grid_sde = sde_params['x_grid'].detach().cpu()
        x_grid_sde = X_grid_sde[input_var] 
        M0_sde = sde_params['M0']
        X_medss = sde_params['x_med']
        x_medss = X_medss[input_var]
        
        if sub_M0: 
            Y_medss = sde_params['y_med']
            y_medss = Y_medss[input_var]

        if sde_cloud == True: 
            y_pred_sde = Y_pred_sde[input_var].detach().cpu() if not sub_M0 else Y_pred_sde[input_var].detach().cpu() - M0_sde
            ax.scatter(x_grid_sde, y_pred_sde, s=18, alpha=0.08, color='C1', label='Diffusion cloud')
        
        ax.plot(x_medss, y_medss, color="#882255", lw=2, label="Diffusion median")

    xlabel = LATEX_LABELS.get(input_var, input_var)
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(r"$\mu - m_B$")
    ax.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels 


### Plot residuals between two sets of y (they must have same axis)





### Plot hubble residuals histogram 
def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _resolve_model_color(label, colors=None, default=None):
    if colors is None:
        colors = {}
    if label in colors:
        return colors[label]

    label_lower = label.lower()
    if "test" in label_lower:
        return "#66CCEE"
    if "sde train" in label_lower:
        return "#882255"
    if "gp train" in label_lower:
        return "#DD8452"

    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    if default is not None:
        return default
    return "C0"

def _resolve_model_linestyle(label, linestyles=None, default="-"):
    if linestyles is None:
        linestyles = {}
    if label in linestyles:
        return linestyles[label]
    if label in TRIPP_LINESTYLES:
        return TRIPP_LINESTYLES[label]
    return default


def _resolve_model_alpha(label, alphas=None, default=0.2):
    if alphas is None:
        return default
    if isinstance(alphas, dict):
        return alphas.get(label, default)
    return alphas


def plot_residual_comparison(
    residuals,
    *,
    colors=None,
    linestyles=None,
    bins=60,
    figsize=(10, 6.5),
    save_fig=False,
    show_fig=True,
    title="Hubble Residuals",
    save_title=None,
    xspace_legend=1.07, 
    yspace_legend=0.98,
    xspace_stats=1.03,
    yspace_title=1.02,
):
    residuals_np = {label: _to_numpy(values) for label, values in residuals.items()}
    if not residuals_np:
        raise ValueError("residuals must contain at least one series.")

    if colors is None:
        colors = {}
    colors = {label: _resolve_model_color(label, colors, default=f"C{i}") for i, label in enumerate(residuals_np)}
    linestyles = {label: _resolve_model_linestyle(label, linestyles, default="-") for label in residuals_np}

    all_values = np.concatenate(list(residuals_np.values()))
    q_lo, q_hi = np.quantile(all_values, [0.01, 0.99])
    span = max(q_hi - q_lo, 1e-6)
    x_lo = min(all_values.min(), q_lo - 0.08 * span)
    x_hi = max(all_values.max(), q_hi + 0.08 * span)
    hist_bins = np.linspace(x_lo, x_hi, bins + 1)

    fig, (ax_hist, ax_box) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.2], "hspace": 0.03},
    )

    for ax in (ax_hist, ax_box):
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    summary = {}
    summary_lines = []

    for label, values in residuals_np.items():
        color = colors[label]
        linestyle = linestyles[label]
        median = float(np.median(values))
        mean = float(np.mean(values))
        std = float(np.std(values))
        rms = float(np.sqrt(np.mean(values**2)))
        mad = float(np.median(np.abs(values - median)))
        # summary[label] = {"median": median, "mean": mean, "std": std, "rms": rms, "mad": mad}
        # summary_lines.append(f"{label}: mean={mean:.3f}, med={median:.3f}, rms={rms:.3f}, std={std:.3f}")

        summary[label] = {"median": median, "mean": mean, "std": std, "mad": mad}
        summary_lines.append(f"{label}: mean={mean:.3f}, med={median:.3f}, std={std:.3f}")

        ax_hist.hist(
            values,
            bins=hist_bins,
            density=False,
            histtype="step",
            alpha=0.18,
            color=color,
            linestyle=linestyle,
        )
        ax_hist.hist(
            values,
            bins=hist_bins,
            density=False,
            histtype="step",
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    ax_hist.axvline(0.0, color="#222222", ls='--', linewidth=1, alpha=0.7)
    ax_hist.set_ylabel("count")
    ax_hist.set_title(title, loc="center", fontsize=15, y=yspace_title)
    ax_hist.legend(
        frameon=False,
        ncol=len(residuals_np),
        loc="upper right",
        bbox_to_anchor=(xspace_legend, yspace_legend),
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.3,
        borderaxespad=0.2,
    )


    ax_hist.text(
        xspace_stats,
        0.85,
        "\n".join(summary_lines),
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
    )

    labels = list(residuals_np.keys())
    values = [residuals_np[label] for label in labels]
    positions = np.arange(len(labels), 0, -1)
    box = ax_box.boxplot(
        values,
        vert=False,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "#555555"},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.3)

    ax_box.axvline(0.0, color="#222222", ls='--', linewidth=1, alpha=0.7)
    ax_box.set_yticks(positions)
    ax_box.set_yticklabels(labels)
    ax_box.set_xlabel(r"$\Delta \mu$")
    ax_box.set_ylabel("")

    plt.tight_layout()

    if save_fig:
        base = '/Users/joygong/CAMBRIDGE/snhost/plots/global/'
        os.makedirs(base, exist_ok=True)
        filename = save_title if save_title is not None else "residual_comparison.pdf"
        plt.savefig(os.path.join(base, filename), bbox_inches='tight')

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_hist, ax_box), summary


def plot_hubble_diagram(
    zcmb,
    model_mu,
    *,
    mu_obs=None,
    mu_obs_alpha=0.8,
    mu_obs_zorder=1,
    colors=None,
    linestyles=None,
    figsize=(10, 6),
    show_points=False,
    scatter_only=False,
    point_alpha=0.2,
    point_size=12,
    ylim=None,
    save_fig=False,
    show_fig=True,
    title="Hubble Diagram",
    save_title=None,
):
    zcmb = _to_numpy(zcmb).reshape(-1)
    if len(zcmb) == 0:
        raise ValueError("zcmb must contain at least one value.")

    if model_mu is None:
        model_mu = {}
    if not model_mu and mu_obs is None:
        raise ValueError("Provide at least one model prediction array or mu_obs.")

    sort_idx = np.argsort(zcmb)
    z_sorted = zcmb[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)

    if mu_obs is not None:
        mu_obs = _to_numpy(mu_obs).reshape(-1)
        if len(mu_obs) != len(zcmb):
            raise ValueError(
                "mu_obs and zcmb must have the same length; "
                f"got {len(mu_obs)} and {len(zcmb)}."
            )
        ax.plot(
            z_sorted,
            mu_obs[sort_idx],
            linewidth=1.8,
            alpha=mu_obs_alpha,
            color="0.35",
            # label="data",
            label="_nolegend_",
            zorder=mu_obs_zorder,
        )

    for label, mu in model_mu.items():
        mu = _to_numpy(mu).reshape(-1)
        if len(mu) != len(zcmb):
            raise ValueError(
                f"Model '{label}' has length {len(mu)} but zcmb has length {len(zcmb)}."
            )

        color = _resolve_model_color(label, colors)
        linestyle = _resolve_model_linestyle(label, linestyles, default="-")
        alpha = _resolve_model_alpha(label, point_alpha, default=0.2)
        mu_sorted = mu[sort_idx]

        if not scatter_only:
            ax.plot(
                z_sorted,
                mu_sorted,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
                zorder=2,
            )

        if show_points or scatter_only:
            ax.scatter(
                z_sorted,
                mu_sorted,
                s=point_size,
                color=color,
                alpha=alpha,
                label=label if scatter_only else None,
                zorder=3 if scatter_only else 2,
            )

    ax.set_xlabel(r"$z_{\rm CMB}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)

    plt.tight_layout()

    if save_fig:
        base = '/Users/joygong/CAMBRIDGE/snhost/plots/'
        os.makedirs(base, exist_ok=True)
        filename = save_title if save_title is not None else "hubble_diagram.pdf"
        plt.savefig(os.path.join(base, filename), bbox_inches='tight')

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


### Calculate stats for hubble residuals and plot them in bins to measure if they are biased as a function of host parameter or redshift 
def quantile_bins(x, n_bins=8):
    x = _to_numpy(x).reshape(-1)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    return edges

def load_tripp_step_thresholds(
    mass_params_path=None,
    mass_ab_params_path=None,
    ssfr_params_path=None,
):
    """
    Step positions from Tripp MCMC: log10(M) uses last tensor element for
    Tripp+mass; Tripp+mass+ab uses ``tracer_thresh`` in the saved dict;
    log10(sSFR) uses last tensor element for Tripp+sSFR.
    """
    if mass_params_path is None:
        mass_params_path = _DEFAULT_TRIPP_PT_DIR / "Tripp_mass_params.pt"
    if mass_ab_params_path is None:
        mass_ab_params_path = _DEFAULT_TRIPP_PT_DIR / "Tripp_mass_ab_params.pt"
    if ssfr_params_path is None:
        ssfr_params_path = _DEFAULT_TRIPP_PT_DIR / "Tripp_sSFR_params.pt"

    mass_params = torch.load(mass_params_path, weights_only=False)
    mass_ab_params = torch.load(mass_ab_params_path, weights_only=False)
    ssfr_params = torch.load(ssfr_params_path, weights_only=False)

    thresh_mass = float(_to_numpy(mass_params[-1]))
    if isinstance(mass_ab_params, dict):
        thresh_massab = float(_to_numpy(mass_ab_params["tracer_thresh"]))
    else:
        thresh_massab = float(_to_numpy(mass_ab_params[-1]))
    thresh_ssfr = float(_to_numpy(ssfr_params[-1]))
    return thresh_mass, thresh_massab, thresh_ssfr


def _residual_step_center(values, stat):
    if stat == "median":
        return float(np.median(values))
    if stat == "mean":
        return float(np.mean(values))
    raise ValueError(f"stat must be 'median' or 'mean' (got {stat!r})")

def _residual_step_error(values, stat):
    """
    Standard error on the central statistic.
    mean   -> std / sqrt(n)
    median -> 1.2533 * std / sqrt(n)   (SE of the median under normality)
    """
    n = len(values)
    if n < 2:
        return float(np.nan)
    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(n)
    if stat == "median":
        return 1.2533 * sem
    return sem

def residual_step_bias(residual, x_host, thresh, stat="median"):
    """
    Central residual with x_host >= thresh minus that with x_host < thresh
    (same low/high convention as ``tracer_meds < tracer_thresh`` in Tripp).

    ``stat`` is ``\"median\"`` or ``\"mean\"`` for each host side.
    """
    residual = _to_numpy(residual).reshape(-1)
    x_host = _to_numpy(x_host).reshape(-1)
    if len(residual) != len(x_host):
        raise ValueError(
            "residual and x_host must have the same length; "
            f"got {len(residual)} and {len(x_host)}."
        )
    low = x_host < thresh
    high = x_host >= thresh
    if not np.any(low) or not np.any(high):
        return np.nan
    return _residual_step_center(residual[high], stat) - _residual_step_center(
        residual[low], stat
    )

# def residual_step_bias(residual, x_host, thresh, stat="median"):
#     """
#     Central residual with x_host >= thresh minus that with x_host < thresh.

#     Returns:
#         step_bias = stat(high) - stat(low)
#         high_stat = stat(residual | high host)
#         low_stat  = stat(residual | low host)
#     """
#     residual = _to_numpy(residual).reshape(-1)
#     x_host = _to_numpy(x_host).reshape(-1)

#     if len(residual) != len(x_host):
#         raise ValueError(
#             "residual and x_host must have the same length; "
#             f"got {len(residual)} and {len(x_host)}."
#         )

#     low = x_host < thresh
#     high = x_host >= thresh

#     if not np.any(low) or not np.any(high):
#         return {
#             "step_bias": np.nan,
#             "low_stat": np.nan,
#             "high_stat": np.nan,
#         }

#     low_stat = _residual_step_center(residual[low], stat)
#     high_stat = _residual_step_center(residual[high], stat)

#     return {
#         "step_bias": high_stat - low_stat,
#         "low_stat": low_stat,
#         "high_stat": high_stat,
#     }

def residual_host_step_bias_table(
    residuals,
    log10M,
    log10sSFR,
    *,
    stat="median",
    thresh_mass=None,
    thresh_massab=None,
    thresh_ssfr=None,
    mass_params_path=None,
    mass_ab_params_path=None,
    ssfr_params_path=None,
):
    """
    One row per entry in ``residuals``: remaining host step contrast at the
    Tripp+mass threshold, Tripp+mass+ab threshold, and Tripp+sSFR threshold.

    Each column is stat(HR | high host) - stat(HR | low host) with
    ``stat`` either ``\"median\"`` or ``\"mean\"``.
    """
    if thresh_mass is None or thresh_massab is None or thresh_ssfr is None:
        tm, tma, ts = load_tripp_step_thresholds(
            mass_params_path=mass_params_path,
            mass_ab_params_path=mass_ab_params_path,
            ssfr_params_path=ssfr_params_path,
        )
        thresh_mass = tm if thresh_mass is None else thresh_mass
        thresh_massab = tma if thresh_massab is None else thresh_massab
        thresh_ssfr = ts if thresh_ssfr is None else thresh_ssfr

    rows = []
    for name, res in residuals.items():
        rows.append(
            {
                "model": name,
                "residual_mass_bias_mass_steppos": residual_step_bias(
                    res, log10M, thresh_mass, stat=stat
                ),
                "residual_mass_bias_massab_steppos": residual_step_bias(
                    res, log10M, thresh_massab, stat=stat
                ),
                "residual_ssfr_bias_ssfr_steppos": residual_step_bias(
                    res, log10sSFR, thresh_ssfr, stat=stat
                ),
            }
        )

    # rows = []

    # for name, res in residuals.items():

    #     mass = residual_step_bias(
    #         res, log10M, thresh_mass, stat=stat
    #     )

    #     massab = residual_step_bias(
    #         res, log10M, thresh_massab, stat=stat
    #     )

    #     ssfr = residual_step_bias(
    #         res, log10sSFR, thresh_ssfr, stat=stat
    #     )

    # rows.append(
    #     {
    #         "model": name,

    #         # mass step
    #         "residual_mass_bias_mass_steppos":
    #             mass["step_bias"],
    #         "mass_low_host_stat":
    #             mass["low_stat"],
    #         "mass_high_host_stat":
    #             mass["high_stat"],

    #         # mass+ab step
    #         "residual_mass_bias_massab_steppos":
    #             massab["step_bias"],
    #         "massab_low_host_stat":
    #             massab["low_stat"],
    #         "massab_high_host_stat":
    #             massab["high_stat"],

    #         # sSFR step
    #         "residual_ssfr_bias_ssfr_steppos":
    #             ssfr["step_bias"],
    #         "ssfr_low_host_stat":
    #             ssfr["low_stat"],
    #         "ssfr_high_host_stat":
    #             ssfr["high_stat"],
    #     }
    # )

    meta = {
        "stat": stat,
        "thresh_log10M_Tripp_mass": thresh_mass,
        "thresh_log10M_Tripp_massab": thresh_massab,
        "thresh_log10sSFR_Tripp_sSFR": thresh_ssfr,
    }

    return pd.DataFrame(rows), meta


def chauvenet_z_thresh(n):                  # how many std away from mean 
    if n < 3:
        return np.nan
    return norm.ppf(1 - 1 / (4 * n))

def chauvenet_mask(values):      # computing chauvenet mask on # pts in bin, not based on whole sample 
    values = _to_numpy(values).reshape(-1)
    n = len(values)
    if n < 3:
        return np.zeros(n, dtype=bool)

    mu = np.mean(values)        # also on mean in the bin, NOT 0 
    sigma = np.std(values, ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        return np.zeros(n, dtype=bool)

    z = np.abs(values - mu) / sigma
    z_thresh = chauvenet_z_thresh(n)
    return z > z_thresh

def _prepare_sample_ids(sample_ids, n_rows):
    if sample_ids is None:
        return np.arange(n_rows)
    sample_ids = _to_numpy(sample_ids).reshape(-1)
    if len(sample_ids) != n_rows:
        raise ValueError(
            f"sample_ids must have length {n_rows}; got {len(sample_ids)}."
        )
    return sample_ids

def binned_residual_stats(x, residual, bins, x_mid_reference=None, sample_ids=None):
    x = _to_numpy(x).reshape(-1)
    residual = _to_numpy(residual).reshape(-1)
    if x_mid_reference is None:
        x_mid_reference = x
    else:
        x_mid_reference = _to_numpy(x_mid_reference).reshape(-1)
    sample_ids = _prepare_sample_ids(sample_ids, len(residual))
    if len(x) != len(residual):
        raise ValueError(
            "x and residual must have the same length; "
            f"got len(x)={len(x)} and len(residual)={len(residual)}."
        )

    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (x >= lo) & (x <= hi)
            ref_mask = (x_mid_reference >= lo) & (x_mid_reference <= hi)
        else:
            mask = (x >= lo) & (x < hi)
            ref_mask = (x_mid_reference >= lo) & (x_mid_reference < hi)

        r = residual[mask]
        xb = x[mask]
        ids_bin = sample_ids[mask]
        x_ref_bin = x_mid_reference[ref_mask]
        n_bin = len(r)
        z_thresh = chauvenet_z_thresh(n_bin)
        x_mid = np.median(x_ref_bin) if len(x_ref_bin) > 0 else 0.5 * (lo + hi)
        chauvenet_bin_mask = chauvenet_mask(r)
        abs_gt_0p3_mask = np.abs(r) > 0.3
        abs_gt_0p5_mask = np.abs(r) > 0.5

        if len(r) == 0:
            rows.append({
                "bin_left": lo,
                "bin_right": hi,
                "x_mid": x_mid,
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "rms": np.nan,
                "std": np.nan,
                "z_thresh": np.nan,
                "n_abs_gt_0p3": 0,
                "n_abs_gt_0p5": 0,
                "n_chauvenet": 0,
                # "abs_gt_0p3_ids": [],
                # "abs_gt_0p5_ids": [],
                "chauvenet_ids": [],
            })
            continue

        rows.append({
            "bin_left": lo,
            "bin_right": hi,
            "x_mid": x_mid,
            "n": len(r),
            "mean": np.mean(r),
            "median": np.median(r),
            "rms": np.sqrt(np.mean(r**2)),
            "std": np.std(r, ddof=1) if len(r) > 1 else np.nan,    # sample std 
            "z_thresh": z_thresh,                                  # threshold for chauvenet
            "n_abs_gt_0p3": np.sum(abs_gt_0p3_mask),
            "n_abs_gt_0p5": np.sum(abs_gt_0p5_mask),
            "n_chauvenet": np.sum(chauvenet_bin_mask),
            # "abs_gt_0p3_ids": ids_bin[abs_gt_0p3_mask].tolist(),
            # "abs_gt_0p5_ids": ids_bin[abs_gt_0p5_mask].tolist(),
            "chauvenet_ids": ids_bin[chauvenet_bin_mask].tolist(),
        })

    return pd.DataFrame(rows)

def _split_residual_spec(model_name, residual_spec, indices_by_model=None):
    values = residual_spec
    indices = None

    if isinstance(residual_spec, dict):
        if "values" not in residual_spec:
            raise ValueError(
                f"Residual spec for model '{model_name}' must contain a 'values' key."
            )
        values = residual_spec["values"]
        indices = residual_spec.get("indices")

    if indices_by_model is not None and model_name in indices_by_model:
        indices = indices_by_model[model_name]

    values = _to_numpy(values).reshape(-1)
    if indices is None:
        return values, None

    indices = _to_numpy(indices).reshape(-1)
    if indices.dtype != np.bool_:
        indices = indices.astype(int)
    return values, indices


def _select_predictor_for_residual(x, residual, indices, *, pred_name, model_name):
    x = _to_numpy(x).reshape(-1)
    if indices is None:
        x_selected = x
    else:
        if indices.dtype == np.bool_ and len(indices) != len(x):
            raise ValueError(
                f"Boolean indices for model '{model_name}' must have the same length "
                f"as predictor '{pred_name}'; got {len(indices)} and {len(x)}."
            )
        x_selected = x[indices]

    if len(x_selected) != len(residual):
        index_hint = " Provide matching indices for this residual." if indices is None else ""
        raise ValueError(
            f"Length mismatch for predictor '{pred_name}' and model '{model_name}': "
            f"selected predictor has {len(x_selected)} rows, "
            f"residual has {len(residual)} rows.{index_hint}"
        )
    return x_selected


def binned_residual_table(
    predictors,
    residuals,
    n_bins=8,
    save_stats=True,
    title_binned=None,
    title_global=None,
    indices_by_model=None,
    sample_ids=None,
): 
    stats_by_predictor = {}
    bins_by_predictor = {}
    all_tables = []
    full_sample_ids = _prepare_sample_ids(sample_ids, len(next(iter(predictors.values()))))

    for pred_name, x in predictors.items():
        bins = quantile_bins(x, n_bins=n_bins)
        bins_by_predictor[pred_name] = bins
        stats_by_predictor[pred_name] = {}

        for model_name, residual_spec in residuals.items():
            res, indices = _split_residual_spec(model_name, residual_spec, indices_by_model)
            x_selected = _select_predictor_for_residual(
                x,
                res,
                indices,
                pred_name=pred_name,
                model_name=model_name,
            )
            sample_ids_selected = full_sample_ids if indices is None else full_sample_ids[indices]
            df = binned_residual_stats(
                x_selected,
                res,
                bins,
                x_mid_reference=x,
                sample_ids=sample_ids_selected,
            )
            df["predictor"] = pred_name
            df["model"] = model_name
            stats_by_predictor[pred_name][model_name] = df
            all_tables.append(df)

    stats_df = pd.concat(all_tables, ignore_index=True)

    # global outlier summary
    global_outliers = []
    for model_name, residual_spec in residuals.items():
        res, indices = _split_residual_spec(model_name, residual_spec, indices_by_model)
        sample_ids_selected = full_sample_ids if indices is None else full_sample_ids[indices]
        global_chauvenet_mask = chauvenet_mask(res)
        global_abs_gt_0p3_mask = np.abs(res) > 0.3
        global_abs_gt_0p5_mask = np.abs(res) > 0.5
        global_outliers.append({
            "model": model_name,
            "N": len(res),
            "mean": np.mean(res),
            "median": np.median(res),
            "rms": np.sqrt(np.mean(res**2)),
            "std": np.std(res, ddof=1),
            "N(|res|>0.3)": np.sum(global_abs_gt_0p3_mask),
            "frac(|res|>0.3)": np.mean(global_abs_gt_0p3_mask),
            "N(|res|>0.5)": np.sum(global_abs_gt_0p5_mask),
            "frac(|res|>0.5)": np.mean(global_abs_gt_0p5_mask),
            "N(Chauvenet)": np.sum(global_chauvenet_mask),
            "frac(Chauvenet)": np.mean(global_chauvenet_mask),
            # "abs_gt_0p3_ids": sample_ids_selected[global_abs_gt_0p3_mask].tolist(),
            # "abs_gt_0p5_ids": sample_ids_selected[global_abs_gt_0p5_mask].tolist(),
            "chauvenet_ids": sample_ids_selected[global_chauvenet_mask].tolist(),
        })

    global_outliers_df = pd.DataFrame(global_outliers)

    if save_stats: 
        base = '/Users/joygong/CAMBRIDGE/snhost/outputs/'
        os.makedirs(base, exist_ok=True)
        title_binned = title_binned or "binned_residual_stats.csv"
        title_global = title_global or "global_outliers.csv"
        stats_df.to_csv(os.path.join(base, 'binned/', title_binned), index=False)
        global_outliers_df.to_csv(os.path.join(base, 'global/', title_global), index=False)

    return stats_df, global_outliers_df, bins_by_predictor


def config_plotting(
    zcmb_cat,
    log10M_remain_med,
    log10sSFR_med,
    train_idx=None,
    test_idx=None,
    hist_indices=None,
): 
    predictors = {
        "redshift": _to_numpy(zcmb_cat).reshape(-1),
        "stellar mass": _to_numpy(log10M_remain_med).reshape(-1),
        "sSFR": _to_numpy(log10sSFR_med).reshape(-1),
    }

    xlabels = {
        "redshift": r"$z_{\rm CMB}$",
        "stellar mass": r"$\log_{10}(M/M_\odot)$",
        "sSFR": r"$\log_{10}(\mathrm{sSFR})$",
    }

    colors = MODEL_COLORS.copy()

    if hist_indices is None:
        hist_indices = {}
        if train_idx is not None:
            hist_indices["train"] = train_idx
        if test_idx is not None:
            hist_indices["test"] = test_idx

    return predictors, xlabels, colors, hist_indices

# Plot binned residuals
def plot_binned_residuals(
    stats_df,
    bins_by_predictor,
    residuals,
    predictors,
    xlabels,
    colors,
    hist_indices=None,
    linestyles=None,
    legend_y=0.03,
    save_fig=False,
    show_fig=True,
    title=None,
): 

    Tripp_sSFR_params = torch.load("Tripp_sSFR_params.pt")
    Tripp_mass_params = torch.load("Tripp_mass_params.pt")
    Tripp_massab_params = torch.load("Tripp_mass_ab_params.pt", weights_only=False)
    Tripp_sSFR_params_steppos = float(_to_numpy(Tripp_sSFR_params[-1]))
    Tripp_mass_params_steppos = float(_to_numpy(Tripp_mass_params[-1]))
    Tripp_massab_params_steppos = float(_to_numpy(Tripp_massab_params['tracer_thresh']))

    predictor_order = ["redshift", "stellar mass", "sSFR"]
    metrics_to_plot = ["median", "rms", "std"]
    model_linestyles = {
        model_name: _resolve_model_linestyle(model_name, linestyles, default="-")
        for model_name in residuals.keys()
    }
    tripp_mass_step_ls = _resolve_model_linestyle("Tripp+mass", linestyles, default=TRIPP_LINESTYLES["Tripp+mass"])
    tripp_massab_step_ls = _resolve_model_linestyle("Tripp+mass+ab", linestyles, default=TRIPP_LINESTYLES["Tripp+mass+ab"])
    tripp_ssfr_step_ls = _resolve_model_linestyle("Tripp+sSFR", linestyles, default=TRIPP_LINESTYLES["Tripp+sSFR"])

    ylabels = {
        "median": r"median($\Delta\mu$)",
        "rms": r"RMS($\Delta\mu$)",
        "std": r"std($\Delta\mu$)",
    }

    fig, axes = plt.subplots(
        5, 3,
        figsize=(14, 15),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [1.0, 1.2, 2.0, 2.0, 2.0], "hspace": 0.05, "wspace": 0.05}
    )

    for col, pred_name in enumerate(predictor_order):
        x = predictors[pred_name]
        qbins = bins_by_predictor[pred_name]
        
        # row 0: histogram
        ax = axes[0, col]

        if hist_indices is None: 
 
            ax.hist(x, bins=28, color="0.85", edgecolor="white", linewidth=0.8)

            for edge in qbins[1:-1]:
                ax.axvline(edge, color="0.55", lw=1.0, alpha=0.8)

            if pred_name == "stellar mass":
                ax.axvline(Tripp_mass_params_steppos, color="#117733", lw=1.5, ls=tripp_mass_step_ls, alpha=0.95, label="Tripp mass step")
                ax.axvline(Tripp_massab_params_steppos, color="#117733", lw=1.5, ls=tripp_massab_step_ls, alpha=0.95, label="Tripp mass+ab step")

            if pred_name == "sSFR":
                ax.axvline(Tripp_sSFR_params_steppos, color="#117733", lw=1.1, ls=tripp_ssfr_step_ls, alpha=0.6, label="Tripp sSFR step")

            if col == 0:
                ax.set_ylabel("count")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
                if pred_name in {"stellar mass", "sSFR"}:
                    ax.legend(frameon=False, loc="upper left", fontsize=10)

            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if hist_indices is not None: 
            hist_bins = np.histogram_bin_edges(x, bins=28)
            
            for split_name, idx in hist_indices.items():
                x_split = x[idx]
                ax.hist(
                    x_split,
                    bins=hist_bins,
                    histtype="step",
                    linewidth=2.0,
                    linestyle=_resolve_model_linestyle(split_name, linestyles, default="-"),
                    color=_resolve_model_color(split_name, colors),
                    density=False,   # use False if you want raw counts instead
                    alpha=0.95,
                    label=split_name if col == 0 else None,
                )
            
            for edge in qbins[1:-1]:
                ax.axvline(edge, color="0.55", lw=1.0, alpha=0.8)

        # row 1: fraction of Chauvenet outliers per bin
        ax = axes[1, col]

        for model_name in residuals.keys():
            df = stats_df[(stats_df['predictor'] == pred_name) & (stats_df['model'] == model_name)]
            frac_chauvenet = 100 * df["n_chauvenet"].to_numpy() / df["n"].to_numpy()
            ax.plot(
                df["x_mid"],
                frac_chauvenet,
                marker="o",
                ms=5,
                lw=2,
                color=_resolve_model_color(model_name, colors),
                linestyle=model_linestyles[model_name],
                alpha=0.95,
                label=model_name if col == 0 else None,
            )

        if pred_name == "stellar mass":
            ax.axvline(Tripp_mass_params_steppos, color="#117733", lw=1.5, ls=tripp_mass_step_ls, alpha=1)
            ax.axvline(Tripp_massab_params_steppos, color="#117733", lw=1.5, ls=tripp_massab_step_ls, alpha=1)

        if pred_name == "sSFR":
            ax.axvline(Tripp_sSFR_params_steppos, color="#117733", lw=1.1, ls=tripp_ssfr_step_ls, alpha=0.65)

        if col == 0:
            ax.set_ylabel("% Chauvenet outliers")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # rows 2-4: median, rms, std
    for row, metric in enumerate(metrics_to_plot, start=2):
        for col, pred_name in enumerate(predictor_order):
            ax = axes[row, col]

            for model_name in residuals.keys():
                df = stats_df[(stats_df['predictor'] == pred_name) & (stats_df['model'] == model_name)]
                ax.plot(
                    df["x_mid"],
                    df[metric],
                    marker="o",
                    ms=5,
                    lw=2,
                    color=_resolve_model_color(model_name, colors),
                    linestyle=model_linestyles[model_name],
                    label=model_name if (row == 4 and col == 0) else None,
                )

            if metric == "median":
                ax.axhline(0.0, color="0.35", lw=1.0, ls="--", alpha=0.9)

            if pred_name == "stellar mass":
                ax.axvline(Tripp_mass_params_steppos, color="#117733", lw=1.5, ls=tripp_mass_step_ls, alpha=0.9)
                ax.axvline(Tripp_massab_params_steppos, color="#117733", lw=1.5, ls=tripp_massab_step_ls, alpha=0.9)

            if pred_name == "sSFR":
                ax.axvline(Tripp_sSFR_params_steppos, color="#117733", lw=1.1, ls=tripp_ssfr_step_ls, alpha=0.6)

            if col == 0:
                ax.set_ylabel(ylabels[metric])
        
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)


            if row == 4:
                ax.set_xlabel(xlabels[pred_name])

            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for r in [0, 1, 2, 3, 4]:
        axes[r, 0].yaxis.set_label_coords(-0.18, 0.5)

    # Keep "best" placement while nudging the search box upward in y.
    axes[4, 0].legend(
        frameon=False,
        loc="best",
        bbox_to_anchor=(0.0, legend_y, 1.0, 1.0),
        bbox_transform=axes[4, 0].transAxes,
    )

    plt.tight_layout()

    if save_fig: 
        base = '/Users/joygong/CAMBRIDGE/snhost/plots/binned/'
        plt.savefig(os.path.join(base, title), bbox_inches='tight')

    if show_fig: 
        plt.show()
    else: 
        plt.close(fig)


## QQ plots
def qq_against_gaussian(values, standardize=True):
    values = _to_numpy(values)
    y = np.sort(values)
    if standardize:
        values = (values - np.mean(values)) / np.std(values, ddof=1)
        y = np.sort(values)

    n = len(y)
    p = (np.arange(1, n + 1) - 0.5) / n     # plotting positions (probabilities)
    x = norm.ppf(p)                         # compute theoretical quantiles w/ inverse cdf 
    return x, y


def qq_against_each_other(xvals, yvals, standardize=False):
    xvals = _to_numpy(xvals)
    yvals = _to_numpy(yvals)

    if standardize:
        xvals = (xvals - np.mean(xvals)) / np.std(xvals, ddof=1)
        yvals = (yvals - np.mean(yvals)) / np.std(yvals, ddof=1)

    xq = np.sort(xvals)
    yq = np.sort(yvals)
    return xq, yq


def plot_qq_residuals(
    residuals,
    *,
    tripp_model=None,
    colors=None,
    pairs=None,
    standardize_against_gaussian=True,
    standardize_pairwise=False,
    figsize=(14, 8),
    save_fig=False,
    show_fig=True,
    title=None,
):
    residuals_np = {name: _to_numpy(values).reshape(-1) for name, values in residuals.items()}
    if not residuals_np:
        raise ValueError("residuals must contain at least one model.")

    if colors is None:
        colors = {}
    resolved_colors = {
        name: _resolve_model_color(name, colors, default=f"C{i}")
        for i, name in enumerate(residuals_np.keys())
    }

    if pairs is None:
        if tripp_model not in residuals_np:
            raise ValueError(
                f"tripp_model='{tripp_model}' not found in residuals keys: {list(residuals_np.keys())}"
            )
        pairs = [(tripp_model, name) for name in residuals_np.keys() if name != tripp_model]

    for name_x, name_y in pairs:
        if name_x not in residuals_np or name_y not in residuals_np:
            raise ValueError(f"pair ({name_x}, {name_y}) contains unknown model label.")

    n_top = len(residuals_np)
    n_bottom = len(pairs)
    n_cols = max(n_top, n_bottom, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    model_names = list(residuals_np.keys())

    for col, name in enumerate(model_names):
        ax = axes[0, col]
        xq, yq = qq_against_gaussian(residuals_np[name], standardize=standardize_against_gaussian)
        ax.scatter(xq, yq, s=18, color=resolved_colors[name], alpha=0.75)
        lo = min(xq.min(), yq.min())
        hi = max(xq.max(), yq.max())
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.35")
        ax.set_title(name)
        ax.set_xlabel("Gaussian quantiles")
        ax.set_ylabel(f"{name} quantiles")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for col in range(n_top, n_cols):
        axes[0, col].set_visible(False)

    for col, (name_x, name_y) in enumerate(pairs):
        ax = axes[1, col]
        xq, yq = qq_against_each_other(
            residuals_np[name_x],
            residuals_np[name_y],
            standardize=standardize_pairwise,
        )
        ax.scatter(xq, yq, s=18, alpha=0.75, color=resolved_colors[name_y])
        lo = min(xq.min(), yq.min())
        hi = max(xq.max(), yq.max())
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.35")
        ax.set_xlabel(f"{name_x} quantiles")
        ax.set_ylabel(f"{name_y} quantiles")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for col in range(n_bottom, n_cols):
        axes[1, col].set_visible(False)

    plt.tight_layout()

    if save_fig:
        path = '/Users/joygong/CAMBRIDGE/snhost/plots/qq/'
        if not title:
            raise ValueError("save_path must be provided when save_fig=True.")
        plt.savefig(os.path.join(path, title), bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, resolved_colors, pairs


##### CALCULATE ROLLING MEDIAN ######
def rolling_stat(x, y, window=None, n_eval=200, min_count=15, stat='median',
                  mode='width'):
    """
    Compute a sliding-window statistic of y vs x, evaluated at `n_eval`
    points spanning the range of x.
 
    mode='width'  : `window` is a fixed x-width (e.g. 0.3 dex in mass).
                    Window membership = points with |x - x_eval| < window/2.
                    Window size in x is constant; number of points per
                    window varies with local data density.
    mode='count'  : `window` is a fixed number of nearest neighbours in x
                    (e.g. window=50). Number of points per window is
                    constant; the x-width varies with local data density.
                    This is the "unequal bin size in x" version, analogous
                    to your binned_statistic but sliding instead of disjoint.
 
    Points are masked out (np.nan) wherever fewer than `min_count` points
    fall in the window, so the curve doesn't extend into sparse/empty
    regions where the statistic is unreliable.
 
    Returns: x_eval, center, err (SEM on the median/mean), n_used
    """
    x = _to_numpy(x).reshape(-1)
    y = _to_numpy(y).reshape(-1)
    order = np.argsort(x)
    x, y = x[order], y[order]
 
    x_eval = np.linspace(x.min(), x.max(), n_eval)
    center = np.full(n_eval, np.nan)
    err = np.full(n_eval, np.nan)
    n_used = np.zeros(n_eval, dtype=int)
 
    if mode == 'width':
        half = window / 2.0
        for i, xe in enumerate(x_eval):
            mask = np.abs(x - xe) < half
            n = mask.sum()
            n_used[i] = n
            if n < min_count:
                continue
            vals = y[mask]
            center[i] = np.median(vals) if stat == 'median' else np.mean(vals)
            sd = np.std(vals, ddof=1)
            sem = sd / np.sqrt(n)
            err[i] = 1.2533 * sem if stat == 'median' else sem
 
    elif mode == 'count':
        k = int(window)
        # for each x_eval point, take the k nearest points in x (by index
        # via searchsorted, then expand symmetrically)
        idx_sorted = np.searchsorted(x, x_eval)
        n = len(x)
        for i, center_idx in enumerate(idx_sorted):
            lo = max(0, center_idx - k // 2)
            hi = min(n, lo + k)
            lo = max(0, hi - k)  # re-clamp if hi hit the upper edge
            vals = y[lo:hi]
            n_used[i] = len(vals)
            if len(vals) < min_count:
                continue
            center[i] = np.median(vals) if stat == 'median' else np.mean(vals)
            sd = np.std(vals, ddof=1)
            sem = sd / np.sqrt(len(vals))
            err[i] = 1.2533 * sem if stat == 'median' else sem
    else:
        raise ValueError("mode must be 'width' or 'count'")
 
    return x_eval, center, err, n_used

##### PLOT ROLLING MEDIAN ######
def plot_rolling_median(ax, residuals, xhost, models, colors, window,
                         mode='width', n_eval=200, min_count=15, stat='median',
                         scatter_model='Diffusion-4 100', band_alpha=0.25,
                         scatter_alpha=0.15, thresh=None, bias_fn=None,
                         model_labels=None, bias_stat='mean',
                         step_label='', text_positions=None,
                         text_fontsize=11):
    """
    Plot a rolling median (or mean) with a shaded SEM band for each model
    in `models`, over a grey background scatter of one model's residuals.
 
    If `thresh` is given (and `bias_fn` is your residual_step_bias-style
    function), each model's legend label also states its two-sided
    residual step bias at that threshold, e.g.
        "gaussian process, Δ(mass) = +0.011 mag"
    so the rolling-median shape and the headline step number both show
    up together in one legend, without drawing the step itself.
 
    model_labels : optional dict mapping entries of `models` to a nicer
    display name (e.g. {'GP-4 100': 'gaussian process'}). Falls back to
    the raw model key if not given.
 
    text_positions : optional dict mapping entries of `models` to an
    (x, y) tuple in axes-fraction coordinates. For any model present in
    this dict, a floating text annotation (same style as the step plot's
    in-panel text) is drawn at that position stating its Δ bias, e.g.
        {'Tripp+sSFR': (0.4, 0.3), 'GP-4 100': (0.15, 0.6)}
    Requires `thresh` and `bias_fn` to also be set, since the text needs
    the computed bias value. Models not in this dict get no floating
    text (legend label is still shown as usual).
 
    step_label : optional string used in the floating text, e.g. 'mass'
    or 'sSFR', to produce "Δ(mass) = +0.011 mag" instead of "Δ = ...".
    """
    ax.scatter(xhost, residuals[scatter_model], color='grey',
               alpha=scatter_alpha, s=15, zorder=0)
 
    delta_str = f"({step_label})" if step_label else ""
 
    for model, color in zip(models, colors):
        x_eval, center, err, n_used = rolling_stat(
            xhost, residuals[model], window=window, n_eval=n_eval,
            min_count=min_count, stat=stat, mode=mode,
        )
 
        display_name = (model_labels or {}).get(model, model)
        label = display_name
        bias = None
        if thresh is not None and bias_fn is not None:
            bias = bias_fn(residuals[model], xhost, thresh=thresh, stat=bias_stat)
            label = f"{display_name}, Δ{delta_str} = {bias['bias']:+.3f} mag"
 
        ax.plot(x_eval, center, color=color, lw=2, label=label, zorder=3)
        ax.fill_between(x_eval, center - err, center + err,
                         color=color, alpha=band_alpha, lw=0, zorder=2)
 
        # optional floating in-panel text, same style as the step plot
        if text_positions is not None and model in text_positions:
            if bias is None:
                raise ValueError(
                    f"text_positions given for '{model}' but thresh/bias_fn "
                    "were not provided, so no bias value is available to "
                    "annotate."
                )
            tx, ty = text_positions[model]
            ax.text(
                tx, ty,
                f"{display_name}\nΔ{delta_str} = {bias['bias']:+.3f} mag",
                transform=ax.transAxes, fontsize=text_fontsize, color=color,
            )
 
    # thin dotted vertical guide at the threshold (visual anchor, not a fit)
    if thresh is not None:
        ax.axvline(thresh, color='lightgrey', lw=0.8, ls=':', zorder=1)

    # thin dotted horizontal guide at hr = 0 
    # ax.axhline(0, color='grey', lw=0.8, ls=':', zorder=2)

    ax.set_ylabel(r'$\Delta \mu \; (\rm mag)$')
    return ax


###### OVERPLOT STEP #######
def _draw_step(ax, bias, thresh, color, lw=2.5, ls='-', label=None,
               alpha=1.0, errorbar_capsize=4, zorder=4,
               error_style='errorbar', band_alpha=0.25):
    """
    Draw a two-level step: flat horizontal segment on each side of `thresh`
    at the side's mean/median, with the SEM shown either as a point error
    bar centered on each segment ('errorbar') or as a shaded band spanning
    the full width of each segment ('band'), and a thin dotted vertical
    guide at the threshold. No diagonal connector between the two levels.
 
    error_style : 'errorbar', 'band', or 'both'
    """
    # horizontal segments (span the actual x-range of points on each side)
    ax.hlines(bias['ylow'], bias['xmin_low'], bias['xmax_low'],
              color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
              label=label)
    ax.hlines(bias['yhigh'], bias['xmin_high'], bias['xmax_high'],
              color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
 
    if error_style in ('errorbar', 'both'):
        # vertical SEM error bars, centered on each segment (placed at
        # xlow/xhigh, the central x value of each side)
        ax.errorbar(bias['xlow'], bias['ylow'], yerr=bias['yerr_low'],
                     fmt='none', ecolor=color, elinewidth=lw * 0.7,
                     capsize=errorbar_capsize, alpha=alpha, zorder=zorder)
        ax.errorbar(bias['xhigh'], bias['yhigh'], yerr=bias['yerr_high'],
                     fmt='none', ecolor=color, elinewidth=lw * 0.7,
                     capsize=errorbar_capsize, alpha=alpha, zorder=zorder)
 
    if error_style in ('band', 'both'):
        # shaded SEM band spanning the full x-range of each segment
        ax.fill_between([bias['xmin_low'], bias['xmax_low']],
                         bias['ylow'] - bias['yerr_low'],
                         bias['ylow'] + bias['yerr_low'],
                         color=color, alpha=band_alpha, lw=0, zorder=zorder - 1)
        ax.fill_between([bias['xmin_high'], bias['xmax_high']],
                         bias['yhigh'] - bias['yerr_high'],
                         bias['yhigh'] + bias['yerr_high'],
                         color=color, alpha=band_alpha, lw=0, zorder=zorder - 1)
 
    # thin dotted vertical guide at the threshold (visual anchor, not a fit)
    ax.axvline(thresh, color='lightgrey', lw=0.8, ls=':', zorder=1)

    # thin dotted horizontal guide at hr = 0 
    ax.axhline(0, color='lightgrey', lw=0.8, ls=':', zorder=2)

def plot_resid_bias(ax, residuals, xhost, xname, masstype, model, error_style='band', band_alpha=0.25, 
                            t_x=0.05, t_y=0.3, m_x=0.05, m_y=0.15):
    if xname == 'mass':
        tripp_model = 'Tripp+sSFR'
        if masstype == 1: 
            thresh = Tripp_mass_params_steppos
        else: 
            thresh = Tripp_massab_params_steppos
        title_text = "corrected for sSFR → bias leaks into mass"
        step_label = "mass"
    else:
        tripp_model = 'Tripp+mass'
        thresh = Tripp_sSFR_params_steppos
        title_text = "corrected for mass → bias leaks into sSFR"
        step_label = "sSFR"
 
    # NOTE: this must call `residual_step_bias` (no leading underscore).
    # If your notebook also has an older `_residual_step_bias` floating
    # around, calling that instead silently gives a dict without
    # 'yerr_low'/'yerr_high'/'n_low' etc., so the error bars won't draw.
    tripp_bias = _residual_step_bias(residuals[tripp_model], xhost, thresh=thresh, stat='mean')
    gp_bias = _residual_step_bias(residuals['GP-4 100'], xhost, thresh=thresh, stat='mean')
    diffusion_bias = _residual_step_bias(residuals['Diffusion-4 100'], xhost, thresh=thresh, stat='mean')
 
    # background scatter (Tripp residuals, for context)
    ax.scatter(xhost, residuals[tripp_model], s=20, color='grey', alpha=0.1, zorder=0)
 
    # Tripp step (grey, dashed)
    _draw_step(ax, tripp_bias, thresh, color='grey', lw=2, ls='--',
               label=f'Tripp residual step, {xname}', error_style=error_style, band_alpha=band_alpha)
 
    # Model step (GP or diffusion, solid, colored)
    if model == 'gp':
        mname, mcolor, mbias = 'gaussian process', '#DD8452', gp_bias
    else:
        mname, mcolor, mbias = 'diffusion', '#882255', diffusion_bias
    _draw_step(ax, mbias, thresh, color=mcolor, lw=2.5, ls='-', error_style=error_style, band_alpha=band_alpha)
 
    ## text
    ax.text(0.0, 1.05, title_text, transform=ax.transAxes, fontsize=11)
 
    # ax.text(
    #     0.4, 0.3,
    #     f"single-parameter step\n"
    #     f"Δ({step_label}) = {tripp_bias['bias']:+.3f} mag",
    #     transform=ax.transAxes, fontsize=11, color='#363535',
    # )
    # ax.text(
    #     0.15, 0.6,
    #     f"{mname}\nΔ({step_label}) = {mbias['bias']:+.3f} mag",
    #     transform=ax.transAxes, fontsize=11, color=mcolor,
    # )
 
    ax.text(
        t_x, t_y,
        f"single-parameter step\n"
        f"Δ({step_label}) = {tripp_bias['bias']:+.3f} mag",
        transform=ax.transAxes, fontsize=11, color='#363535',
    )
    ax.text(
        m_x, m_y,
        f"{mname}\nΔ({step_label}) = {mbias['bias']:+.3f} mag",
        transform=ax.transAxes, fontsize=11, color=mcolor,
    )

    if xname == 'mass':
        ax.set_xlabel(r'host stellar mass $\log_{10}(M/M_\odot)$')
    else:
        ax.set_xlabel(r'host sSFR $\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$')
    ax.set_ylabel(r'$\Delta \mu \; (\rm mag)$')