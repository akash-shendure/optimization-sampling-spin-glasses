"""Preliminary scaling study: Ising ferromagnet only, finite-size sweep.

Four diagnostics vs inverse temperature, five lattice sizes, error bars from
disorder realizations.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.models.ising2d import IsingFerromagnet2D
from spinglass.experiments.studies import sampling_beta_sweep
from spinglass.experiments.benchmarks import summarize_sampling_table
from spinglass.experiments.overlap import summarize_replica_overlaps
from spinglass.experiments.budget import sweeps
from spinglass.plotting.style import set_publication_style


def log(msg=""):
    print(msg, flush=True)


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

N_CHAINS = 4
TRACE_EVERY = 10
BUDGET = sweeps(200)
D = 100
N_BETAS = 40

BETAS = np.round(np.linspace(0.05, 1.00, N_BETAS), 4).tolist()
SIZES = [6, 8, 10, 12, 16]
MODEL_NAME = "2D Ising Ferromagnet"


def compute_disorder_stats(result):
    per_disorder = summarize_sampling_table(
        result["table"], records=result["records"],
        group_by=["algorithm_beta", "disorder_id"],
    )
    by_beta = defaultdict(list)
    for row in per_disorder:
        by_beta[row["algorithm_beta"]].append(row)

    metrics = [
        "mean_energy", "mean_acceptance_rate",
        "energy_ess", "energy_rhat", "energy_tau_int",
    ]
    out = []
    for beta in sorted(by_beta.keys()):
        rows = by_beta[beta]
        agg = {"beta": beta}
        for key in metrics:
            vals = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            agg[f"{key}_mean"] = float(np.nanmean(vals)) if valid.size else np.nan
            agg[f"{key}_std"] = float(np.nanstd(vals)) if valid.size > 1 else 0.0
        out.append(agg)
    return out


def compute_disorder_overlap_stats(result):
    per_disorder = summarize_replica_overlaps(
        result["records"], group_by=["algorithm_beta", "disorder_id"]
    )
    if not per_disorder:
        return []
    by_beta = defaultdict(list)
    for row in per_disorder:
        by_beta[row["algorithm_beta"]].append(row)
    out = []
    for beta in sorted(by_beta.keys()):
        rows = by_beta[beta]
        abs_qs = np.array([r.get("mean_abs_q", np.nan) for r in rows], dtype=np.float64)
        valid = abs_qs[~np.isnan(abs_qs)]
        out.append({
            "beta": beta,
            "mean_abs_q_mean": float(np.nanmean(abs_qs)) if valid.size else np.nan,
            "mean_abs_q_std": float(np.nanstd(abs_qs)) if valid.size > 1 else 0.0,
        })
    return out


def _dress_four_panels(axes):
    FS = 11.5  # unified font size for legend, title, tick labels

    axes[0, 0].set_xlabel(r"$\beta$")
    axes[0, 0].set_ylabel("acceptance rate")
    axes[0, 0].set_title("Acceptance rate")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=FS)

    axes[0, 1].set_xlabel(r"$\beta$")
    axes[0, 1].set_ylabel("ESS")
    axes[0, 1].set_title("Effective sample size")
    axes[0, 1].set_yscale("log")

    axes[1, 0].set_xlabel(r"$\beta$")
    axes[1, 0].set_ylabel(r"$\hat{\tau}_{\rm int}$")
    axes[1, 0].set_title(r"Integrated autocorrelation time")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set_xlabel(r"$\beta$")
    axes[1, 1].set_ylabel(r"$\langle |q| \rangle$")
    axes[1, 1].set_title(r"Mean replica overlap $|q|$")
    axes[1, 1].set_ylim(-0.05, 1.05)

    for ax in axes.ravel():
        ax.tick_params(labelsize=FS)


def _plot_four(axes, stats, overlap_stats, color, marker, label, show_eb=True):
    ms = 4
    capsize = 2.5 if show_eb else 0
    fmt = f"{marker}-" if show_eb else "-"
    betas = np.array([r["beta"] for r in stats])

    def _v(key):
        return np.array([r.get(f"{key}_mean", np.nan) for r in stats])

    def _e(key):
        if not show_eb:
            return None
        return np.array([r.get(f"{key}_std", 0.0) for r in stats])

    m = _v("mean_acceptance_rate")
    if not np.all(np.isnan(m)):
        axes[0, 0].errorbar(betas, m, yerr=_e("mean_acceptance_rate"),
                            fmt=fmt, color=color, capsize=capsize,
                            markersize=ms, label=label)
    axes[0, 1].errorbar(betas, _v("energy_ess"), yerr=_e("energy_ess"),
                        fmt=fmt, color=color, capsize=capsize,
                        markersize=ms, label=label)
    axes[1, 0].errorbar(betas, _v("energy_tau_int"), yerr=_e("energy_tau_int"),
                        fmt=fmt, color=color, capsize=capsize,
                        markersize=ms, label=label)
    if overlap_stats:
        bq = np.array([r["beta"] for r in overlap_stats])
        mq = np.array([r["mean_abs_q_mean"] for r in overlap_stats])
        sq = np.array([r["mean_abs_q_std"] for r in overlap_stats]) if show_eb else None
        axes[1, 1].errorbar(bq, mq, yerr=sq, fmt=fmt, color=color,
                            capsize=capsize, markersize=ms, label=label)


def _size_colors(n):
    return [plt.cm.plasma(x) for x in np.linspace(0.12, 0.82, n)]


def main():
    set_publication_style()
    t_global = time.time()
    log("Preliminary scaling analysis (Ising ferromagnet)")
    log(f"  d={D}, {N_BETAS} betas, sweeps={BUDGET.value:.0f}")
    log("=" * 60)

    scaling = {}
    for L in SIZES:
        tag = f"L={L}"
        log(f"\n  [{tag}]")
        t0 = time.time()
        result = sampling_beta_sweep(
            model_class=IsingFerromagnet2D,
            model_kwargs={"L": [L]},
            betas=BETAS,
            n_chains=N_CHAINS,
            n_steps=2000,
            burn_in=500,
            trace_every=TRACE_EVERY,
            n_disorders=D,
            budget=BUDGET,
        )
        log(f"    done in {time.time() - t0:.1f}s")
        scaling[L] = {"discrete": result, "n_spins": L * L}

    log("\n  Generating figure ...")
    colors = _size_colors(len(SIZES))
    all_stats = []
    for L in SIZES:
        data = scaling[L]
        stats = compute_disorder_stats(data["discrete"])
        ov = compute_disorder_overlap_stats(data["discrete"])
        all_stats.append((L, stats, ov))

    for eb in [False, True]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 5.625), constrained_layout=True)
        fig.suptitle(MODEL_NAME, fontsize=14)
        for idx, (L, stats, ov) in enumerate(all_stats):
            _plot_four(axes, stats, ov, colors[idx], "o", f"$n={L * L}$", show_eb=eb)
        _dress_four_panels(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"preliminary_scaling{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)

    elapsed = time.time() - t_global
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("Done.")


if __name__ == "__main__":
    main()
