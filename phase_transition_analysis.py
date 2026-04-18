"""Phase transition analysis: sampling diagnostics vs inverse temperature.

Two figure types per model (8 total), each saved with and without error bars:
  1. Discrete vs relaxed at one system size  (2x3)
  2. Finite-size scaling across 5 sizes      (2x3)

Six diagnostics:
  (a) Mean energy per spin       (d) Split-R-hat
  (b) Acceptance rate            (e) Integrated autocorrelation time
  (c) ESS                        (f) Mean replica overlap |q|

Error bars are standard deviation across disorder realizations.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.models.ising2d import IsingFerromagnet2D
from spinglass.models.edwards_anderson import EdwardsAnderson2D
from spinglass.models.sparse_glass import SparseRandomGlass
from spinglass.models.sk import SherringtonKirkpatrick
from spinglass.experiments.studies import sampling_beta_sweep, relaxed_sampling_beta_sweep
from spinglass.experiments.benchmarks import summarize_sampling_table
from spinglass.experiments.budget import sweeps
from spinglass.plotting.style import set_publication_style, PANEL_COLORS


def log(msg=""):
    print(msg, flush=True)


# ── output ───────────────────────────────────────────────────────────
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── shared parameters ────────────────────────────────────────────────
N_CHAINS = 4
TRACE_EVERY = 10
BUDGET = sweeps(200)
D = 20
N_BETAS = 40

# ── beta grids (50 uniformly spaced points) ──────────────────────────
BETAS_ISING  = np.round(np.linspace(0.05, 1.00, N_BETAS), 4).tolist()
BETAS_EA     = np.round(np.linspace(0.20, 4.00, N_BETAS), 4).tolist()
BETAS_SPARSE = np.round(np.linspace(0.25, 5.00, N_BETAS), 4).tolist()
BETAS_SK     = np.round(np.linspace(0.15, 3.00, N_BETAS), 4).tolist()

# ── model configs ────────────────────────────────────────────────────
# Single-size choices:
#   Ising L=12: above-middle (L=10 is middle of [6..16]) for sharper
#               transition signal at beta_c ~ 0.44
#   EA L=10:    middle of [6..14]
#   Sparse n=72: middle of [32..128]
#   SK n=48:    middle of [24..80]
#
# Scaling sizes:
#   Ising  L = 6,8,10,12,16  (n = 36..256) — standard lattice sizes
#   EA     L = 6,8,10,12,14  (n = 36..196) — same, capped for cost
#   Sparse n = 32,48,72,96,128 — roughly 1.4x geometric spacing
#   SK     n = 24,32,48,64,80  — spans factor 3.3x around beta_c=1

MODELS = {
    "ising2d": dict(
        model_class=IsingFerromagnet2D,
        single_kwargs={"L": [12]},
        size_key="L",
        sizes=[6, 8, 10, 12, 16],
        extra_kwargs={},
        betas=BETAS_ISING,
        model_name="2D Ising Ferromagnet",
        single_params=r"$n=144$",
    ),
    "ea2d": dict(
        model_class=EdwardsAnderson2D,
        single_kwargs={"L": [10], "disorder": ["pm1"]},
        size_key="L",
        sizes=[6, 8, 10, 12, 14],
        extra_kwargs={"disorder": ["pm1"]},
        betas=BETAS_EA,
        model_name="2D Edwards-Anderson",
        single_params=r"$n=100$",
    ),
    "sparse": dict(
        model_class=SparseRandomGlass,
        single_kwargs={"n": [72], "c": [3.0], "disorder": ["gaussian"]},
        size_key="n",
        sizes=[32, 48, 72, 96, 128],
        extra_kwargs={"c": [3.0], "disorder": ["gaussian"]},
        betas=BETAS_SPARSE,
        model_name="Sparse Random Glass",
        single_params=r"$n=72$, $c=3$",
    ),
    "sk": dict(
        model_class=SherringtonKirkpatrick,
        single_kwargs={"n": [48]},
        size_key="n",
        sizes=[24, 32, 48, 64, 80],
        extra_kwargs={},
        betas=BETAS_SK,
        model_name="Sherrington-Kirkpatrick",
        single_params=r"$n=48$",
    ),
}


def _make_title(cfg, scaling=False):
    if scaling:
        return f"{cfg['model_name']} ($d={D}$)"
    return f"{cfg['model_name']} ({cfg['single_params']}, $d={D}$)"


# ═════════════════════════════════════════════════════════════════════
#  per-disorder diagnostic computation
# ═════════════════════════════════════════════════════════════════════

def compute_disorder_stats(result):
    per_disorder = summarize_sampling_table(
        result["table"], records=result["records"],
        group_by=["algorithm_beta", "disorder_id"]
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
        agg = {"beta": beta, "n_disorders": len(rows)}
        for key in metrics:
            vals = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            agg[f"{key}_mean"] = float(np.nanmean(vals)) if valid.size else np.nan
            agg[f"{key}_std"] = float(np.nanstd(vals)) if valid.size > 1 else 0.0
        out.append(agg)
    return out


def compute_disorder_overlap_stats(result):
    overlap_rows = result.get("overlap", [])
    if not overlap_rows:
        return []
    by_beta = defaultdict(list)
    for row in overlap_rows:
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


# ═════════════════════════════════════════════════════════════════════
#  experiment runners
# ═════════════════════════════════════════════════════════════════════

def _run_discrete(model_class, model_kwargs, betas, d, tag=""):
    log(f"    Metropolis {tag} ({len(betas)} betas, d={d}) ...")
    t0 = time.time()
    result = sampling_beta_sweep(
        model_class=model_class, model_kwargs=model_kwargs,
        betas=betas, n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=d, budget=BUDGET,
    )
    log(f"      done in {time.time() - t0:.1f}s  ({len(result['grouped'])} conditions)")
    return result


def _run_relaxed(model_class, model_kwargs, betas, d, tag=""):
    log(f"    Langevin {tag} ({len(betas)} betas, d={d}) ...")
    t0 = time.time()
    result = relaxed_sampling_beta_sweep(
        model_class=model_class, model_kwargs=model_kwargs,
        betas=betas, n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=d, budget=BUDGET,
    )
    log(f"      done in {time.time() - t0:.1f}s  ({len(result['grouped'])} conditions)")
    return result


# ═════════════════════════════════════════════════════════════════════
#  plotting
# ═════════════════════════════════════════════════════════════════════

def _n_spins(cfg):
    if "L" in cfg["single_kwargs"]:
        return cfg["single_kwargs"]["L"][0] ** 2
    return cfg["single_kwargs"]["n"][0]


def _dress_six_panels(axes):
    axes[0, 0].set_xlabel(r"$\beta$")
    axes[0, 0].set_ylabel(r"$\langle H \rangle / n$")
    axes[0, 0].set_title("(a) Mean energy per spin")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel(r"$\beta$")
    axes[0, 1].set_ylabel("acceptance rate")
    axes[0, 1].set_title("(b) Acceptance rate")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].set_xlabel(r"$\beta$")
    axes[0, 2].set_ylabel("ESS")
    axes[0, 2].set_title("(c) Effective sample size")
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].set_xlabel(r"$\beta$")
    axes[1, 0].set_ylabel(r"$\hat{R}$")
    axes[1, 0].set_title(r"(d) Split-$\hat{R}$")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel(r"$\beta$")
    axes[1, 1].set_ylabel(r"$\hat{\tau}_{\rm int}$")
    axes[1, 1].set_title(r"(e) Integrated autocorrelation time")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].set_xlabel(r"$\beta$")
    axes[1, 2].set_ylabel(r"$\langle |q| \rangle$")
    axes[1, 2].set_title(r"(f) Mean replica overlap $|q|$")
    axes[1, 2].set_ylim(-0.05, 1.05)
    axes[1, 2].legend(fontsize=8)


def _plot_six(axes, stats, overlap_stats, n_spins, color, marker, label,
              show_eb=True):
    """Draw one series onto 6 panels. If show_eb=False, suppress error bars."""
    ms = 4
    capsize = 2.5 if show_eb else 0
    betas = np.array([r["beta"] for r in stats])

    def _v(key):
        return np.array([r.get(f"{key}_mean", np.nan) for r in stats])

    def _e(key):
        if not show_eb:
            return None
        return np.array([r.get(f"{key}_std", 0.0) for r in stats])

    # (a) mean energy / n
    axes[0, 0].errorbar(betas, _v("mean_energy") / n_spins,
                        yerr=_e("mean_energy") / n_spins if show_eb else None,
                        fmt=f"{marker}-", color=color, capsize=capsize,
                        markersize=ms, label=label)
    # (b) acceptance rate
    m = _v("mean_acceptance_rate")
    if not np.all(np.isnan(m)):
        axes[0, 1].errorbar(betas, m, yerr=_e("mean_acceptance_rate"),
                            fmt=f"{marker}-", color=color, capsize=capsize,
                            markersize=ms, label=label)
    # (c) ESS
    axes[0, 2].errorbar(betas, _v("energy_ess"), yerr=_e("energy_ess"),
                        fmt=f"{marker}-", color=color, capsize=capsize,
                        markersize=ms, label=label)
    # (d) R-hat
    axes[1, 0].errorbar(betas, _v("energy_rhat"), yerr=_e("energy_rhat"),
                        fmt=f"{marker}-", color=color, capsize=capsize,
                        markersize=ms, label=label)
    # (e) tau_int
    axes[1, 1].errorbar(betas, _v("energy_tau_int"), yerr=_e("energy_tau_int"),
                        fmt=f"{marker}-", color=color, capsize=capsize,
                        markersize=ms, label=label)
    # (f) overlap
    if overlap_stats:
        bq = np.array([r["beta"] for r in overlap_stats])
        mq = np.array([r["mean_abs_q_mean"] for r in overlap_stats])
        sq = np.array([r["mean_abs_q_std"] for r in overlap_stats]) if show_eb else None
        axes[1, 2].errorbar(bq, mq, yerr=sq, fmt=f"{marker}-", color=color,
                            capsize=capsize, markersize=ms, label=label)


# ═════════════════════════════════════════════════════════════════════
#  figure generators (each saves with and without error bars)
# ═════════════════════════════════════════════════════════════════════

def _size_colors(n):
    return [plt.cm.plasma(x) for x in np.linspace(0.12, 0.82, n)]


def plot_diagnostics(name, disc, relax, cfg):
    n_sp = _n_spins(cfg)
    d_stats = compute_disorder_stats(disc)
    d_ov = compute_disorder_overlap_stats(disc)
    r_stats = compute_disorder_stats(relax)
    r_ov = compute_disorder_overlap_stats(relax)

    FIG_DIR.mkdir(exist_ok=True)
    for eb in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(_make_title(cfg), fontsize=14)
        _plot_six(axes, d_stats, d_ov, n_sp,
                  PANEL_COLORS["discrete_sampling"], "o", "Metropolis", show_eb=eb)
        _plot_six(axes, r_stats, r_ov, n_sp,
                  PANEL_COLORS["relaxed_sampling"], "s", "Langevin", show_eb=eb)
        _dress_six_panels(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"diagnostics_{name}{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)


def plot_scaling(name, scaling_data, cfg):
    sizes = cfg["sizes"]
    colors = _size_colors(len(sizes))

    # precompute stats for all sizes
    all_stats = []
    for idx, sz in enumerate(sizes):
        data = scaling_data[sz]
        stats = compute_disorder_stats(data["discrete"])
        ov = compute_disorder_overlap_stats(data["discrete"])
        all_stats.append((sz, data["n_spins"], stats, ov))

    FIG_DIR.mkdir(exist_ok=True)
    for eb in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(_make_title(cfg, scaling=True), fontsize=14)
        for idx, (sz, n_sp, stats, ov) in enumerate(all_stats):
            _plot_six(axes, stats, ov, n_sp, colors[idx], "o",
                      f"$n={n_sp}$", show_eb=eb)
        _dress_six_panels(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"scaling_{name}{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════

def main():
    set_publication_style()
    t_global = time.time()
    log("Phase transition analysis")
    log(f"  d={D}, {N_BETAS} betas, sweeps={BUDGET.value:.0f}")
    log("=" * 60)

    # ── diagnostics: single size, discrete vs relaxed ────────────
    log(f"\n--- Diagnostics (discrete vs relaxed) ---")
    t1 = time.time()

    single = {}
    for name, cfg in MODELS.items():
        log(f"\n  [{name}]")
        mc, mk = cfg["model_class"], cfg["single_kwargs"]
        disc = _run_discrete(mc, mk, cfg["betas"], D)
        relax = _run_relaxed(mc, mk, cfg["betas"], D)
        single[name] = (disc, relax)

    log(f"\n  Diagnostics experiments: {time.time() - t1:.1f}s")
    log("  Generating diagnostic figures ...")
    for name, cfg in MODELS.items():
        plot_diagnostics(name, single[name][0], single[name][1], cfg)

    # ── scaling: 5 sizes, discrete only ──────────────────────────
    log(f"\n--- Scaling (5 sizes per model) ---")
    t2 = time.time()

    scaling = {}
    for name, cfg in MODELS.items():
        sk = cfg["size_key"]
        scaling[name] = {}
        for sz in cfg["sizes"]:
            mk = {sk: [sz]}
            mk.update(cfg["extra_kwargs"])
            tag = f"{sk}={sz}"
            log(f"\n  [{name}] {tag}")
            disc = _run_discrete(cfg["model_class"], mk, cfg["betas"], D, tag)
            n_sp = sz * sz if sk == "L" else sz
            scaling[name][sz] = {"discrete": disc, "n_spins": n_sp}

    log(f"\n  Scaling experiments: {time.time() - t2:.1f}s")
    log("  Generating scaling figures ...")
    for name, cfg in MODELS.items():
        plot_scaling(name, scaling[name], cfg)

    elapsed = time.time() - t_global
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"All figures saved to {FIG_DIR.resolve()}")
    log("Done.")


if __name__ == "__main__":
    main()
