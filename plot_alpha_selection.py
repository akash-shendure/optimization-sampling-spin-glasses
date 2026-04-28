# focused alpha-selection plot: relaxed-sampler diagnostics at alpha in {1, 2, 4} on Ising L=14
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.plotting.style import set_publication_style

CACHE_DIR = Path("results/alpha_sweep")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

SAMPLERS = ["Langevin", "MALA", "HMC"]
ALPHAS_SHOWN = [1.0, 2.0, 4.0]  # why: bracket the alpha=2.0 default chosen by the full sweep

# load a single cached (sampler, alpha) result bundle produced by the alpha sweep
def _load(name, alpha):
    with (CACHE_DIR / f"{name}_alpha{alpha}.pkl").open("rb") as f:
        return pickle.load(f)

# axis labels / titles / scales for the 2x2 diagnostic panel
def _dress(axes):
    fs = 11.5
    axes[0, 0].set_xlabel(r"$\beta$"); axes[0, 0].set_ylabel(r"$\hat{R}$")
    axes[0, 0].set_title(r"Split-$\hat{R}$")
    axes[0, 1].set_xlabel(r"$\beta$"); axes[0, 1].set_ylabel("ESS")
    axes[0, 1].set_title("Effective sample size"); axes[0, 1].set_yscale("log")
    axes[1, 0].set_xlabel(r"$\beta$"); axes[1, 0].set_ylabel(r"$\hat{\tau}_{\rm int}$")
    axes[1, 0].set_title("Integrated autocorrelation time"); axes[1, 0].set_yscale("log")
    axes[1, 1].set_xlabel(r"$\beta$"); axes[1, 1].set_ylabel(r"$\langle |q| \rangle$")
    axes[1, 1].set_title(r"Mean replica overlap $|q|$"); axes[1, 1].set_ylim(-0.05, 1.05)
    for ax in axes.ravel():
        ax.tick_params(labelsize=fs)
        ax.grid(False)

# draw one (sampler, alpha) bundle on the four diagnostic panels
def _plot(axes, bundle, color, ls, label):
    stats = bundle["stats"]; ov = bundle["overlap"]
    betas = np.array([r["beta"] for r in stats])
    rh = np.array([r["energy_rhat_mean"] for r in stats])
    if not np.all(np.isnan(rh)):
        axes[0, 0].plot(betas, rh, ls=ls, color=color, label=label, lw=1.6)
    axes[0, 1].plot(betas, [r["energy_ess_mean"] for r in stats], ls=ls, color=color, label=label, lw=1.6)
    axes[1, 0].plot(betas, [r["energy_tau_int_mean"] for r in stats], ls=ls, color=color, label=label, lw=1.6)
    if ov:
        bq = np.array([r["beta"] for r in ov])
        mq = np.array([r["mean_abs_q_mean"] for r in ov])
        axes[1, 1].plot(bq, mq, ls=ls, color=color, label=label, lw=1.6)

# top-level driver: load cached bundles, build the figure, save to disk
def main():
    set_publication_style()
    # encode sampler with color, alpha with linestyle; one legend covers both axes
    sampler_color = {n: c for n, c in zip(SAMPLERS, plt.cm.tab10(np.arange(len(SAMPLERS))))}
    alpha_styles = {1.0: "-", 2.0: "--", 4.0: ":"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), constrained_layout=True)
    fig.suptitle(r"Ising L=14: relaxed samplers at selected $\alpha$ (linear reg, $\lambda=0$)",
                 fontsize=13)
    # overlay all 9 (sampler, alpha) curves
    for alpha in ALPHAS_SHOWN:
        for name in SAMPLERS:
            label = f"{name}, $\\alpha={alpha}$"
            _plot(axes, _load(name, alpha), sampler_color[name], alpha_styles[alpha], label)
    _dress(axes)
    # 3-column legend keeps the 9-entry block compact in the overlap panel
    leg = axes[1, 1].legend(ncol=3, loc="upper center", frameon=True,
                            framealpha=0.92, columnspacing=1.2,
                            handletextpad=0.6, handlelength=2.0,
                            borderpad=0.4)
    leg.get_frame().set_edgecolor("0.4")

    path = FIG_DIR / "alpha_selection_ising.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    print(f"saved {path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
