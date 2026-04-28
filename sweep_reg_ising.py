# sweep linear vs quartic non-binary regularization with a lambda ladder on Ising L=14
from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.models.ising2d import IsingFerromagnet2D
from spinglass.samplers.langevin import LangevinSampler
from spinglass.samplers.mala import MALASampler
from spinglass.samplers.hmc import HMCSampler
from spinglass.experiments.studies import sampling_beta_sweep
from spinglass.experiments.benchmarks import summarize_sampling_table
from spinglass.experiments.overlap import summarize_replica_overlaps
from spinglass.experiments.budget import sweeps
from spinglass.plotting.style import set_publication_style

# flush=True so progress is visible when piped to a log file
def log(msg=""):
    print(msg, flush=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("results/reg_sweep")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

L = 14  # 196 spins; matches the alpha sweep so figures are comparable
N_CHAINS = 4
TRACE_EVERY = 10
BUDGET = sweeps(200)  # fixed compute per cell so cost comparisons are fair
D = 20
N_BETAS = 30
ALPHA = 1.0  # hold alpha at the canonical pre-sweep value while reg/lambda vary

# beta range crosses the 2d Ising critical point at ~0.4407
BETAS = np.round(np.linspace(0.05, 1.00, N_BETAS), 4).tolist()
LAMBDAS = [0.0, 0.1, 0.5, 1.0]  # 0 is the no-penalty baseline
REGS = ["linear", "quartic"]  # 1-t^2 vs (1-t^2)^2; quartic vanishes faster near +-1
SAMPLERS = [
    ("Langevin", LangevinSampler, {"step_size": 0.02}),
    ("MALA", MALASampler, {"step_size": 0.02}),
    ("HMC", HMCSampler, {"step_size": 0.02, "n_leapfrog": 10}),
]

# aggregate per-disorder rows into per-beta mean/std (no energy metric needed here)
def _disorder_stats(result):
    per_d = summarize_sampling_table(
        result["table"], records=result["records"],
        group_by=["algorithm_beta", "disorder_id"],
    )
    by_beta = defaultdict(list)
    for row in per_d:
        by_beta[row["algorithm_beta"]].append(row)
    metrics = ["mean_acceptance_rate", "energy_ess", "energy_rhat", "energy_tau_int"]
    out = []
    for beta in sorted(by_beta.keys()):
        rows = by_beta[beta]
        agg = {"beta": beta}
        for k in metrics:
            vals = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if valid.size else np.nan
            agg[f"{k}_std"] = float(np.nanstd(vals)) if valid.size > 1 else 0.0
        out.append(agg)
    return out

# aggregate replica-overlap |q| rows the same way as _disorder_stats
def _disorder_overlap_stats(result):
    per_d = summarize_replica_overlaps(
        result["records"], group_by=["algorithm_beta", "disorder_id"]
    )
    if not per_d:
        return []
    by_beta = defaultdict(list)
    for row in per_d:
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

# run one (sampler, reg, lambda) beta sweep on Ising, cached by name+reg+lambda
def run_one(name, cls, skw, reg, lam):
    cache_path = CACHE_DIR / f"{name}_{reg}_lam{lam}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    t0 = time.time()
    result = sampling_beta_sweep(
        model_class=IsingFerromagnet2D,
        model_kwargs={"L": [L]},
        betas=BETAS,
        sampler_class=cls, sampler_kwargs=skw, space="relaxed",
        n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=D, budget=BUDGET,
        alpha=ALPHA, lam=float(lam), reg=reg,
    )
    stats = _disorder_stats(result)
    ov = _disorder_overlap_stats(result)
    bundle = {"stats": stats, "overlap": ov, "lam": lam, "reg": reg, "name": name}
    with cache_path.open("wb") as f:
        pickle.dump(bundle, f)
    log(f"    {name} reg={reg} lam={lam}: done in {time.time()-t0:.1f}s")
    return bundle

# attach axis labels, titles, and log scales to the 2x2 diagnostics grid
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

# overlay one (sampler, lambda) curve set onto the four diagnostic axes
def _plot(axes, bundle, color, ls, label):
    stats = bundle["stats"]; ov = bundle["overlap"]
    betas = np.array([r["beta"] for r in stats])
    rh = np.array([r["energy_rhat_mean"] for r in stats])
    # skip rhat panel if every entry is NaN
    if not np.all(np.isnan(rh)):
        axes[0, 0].plot(betas, rh, ls=ls, color=color, label=label, lw=1.4)
    axes[0, 1].plot(betas, [r["energy_ess_mean"] for r in stats], ls=ls, color=color, label=label, lw=1.4)
    axes[1, 0].plot(betas, [r["energy_tau_int_mean"] for r in stats], ls=ls, color=color, label=label, lw=1.4)
    if ov:
        bq = np.array([r["beta"] for r in ov])
        mq = np.array([r["mean_abs_q_mean"] for r in ov])
        axes[1, 1].plot(bq, mq, ls=ls, color=color, label=label, lw=1.4)

# build the per-reg comparison figure: color = sampler, linestyle = lambda
def figure_one_reg(reg, results):
    sampler_color = {n: c for n, c in zip(
        [s[0] for s in SAMPLERS], plt.cm.tab10(np.arange(len(SAMPLERS))))}
    lam_styles = {l: ls for l, ls in zip(LAMBDAS, ["-", "--", "-.", ":"])}

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    fig.suptitle(rf"Ising L={L}: $\lambda$ sweep (reg={reg}, $\alpha$={ALPHA})", fontsize=14)
    for lam in LAMBDAS:
        for name, _cls, _skw in SAMPLERS:
            label = f"{name}, $\\lambda$={lam}"
            _plot(axes, results[(name, reg, lam)], sampler_color[name], lam_styles[lam], label)
    _dress(axes)
    axes[0, 1].legend(fontsize=7, ncol=3, loc="lower left")  # ESS panel has most room
    path = FIG_DIR / f"reg_sweep_ising_{reg}.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    log(f"  saved {path}")
    plt.close(fig)

# top-level driver: sweep all (reg, lambda, sampler) cells then build one figure per reg
def main():
    set_publication_style()
    t0 = time.time()
    log(f"Reg+lambda sweep on Ising L={L}")
    log(f"  regs={REGS}, lambdas={LAMBDAS}, betas={N_BETAS}, D={D}, sweeps={BUDGET.value:.0f}, alpha={ALPHA}")
    log("=" * 60)

    # sweep loop over (reg, lambda, sampler)
    results = {}
    for reg in REGS:
        for lam in LAMBDAS:
            for name, cls, skw in SAMPLERS:
                log(f"\n  [{name}] reg={reg} lam={lam}")
                results[(name, reg, lam)] = run_one(name, cls, skw, reg, lam)

    # figure assembly: one figure per regularization form
    log("\n  Generating figures ...")
    for reg in REGS:
        figure_one_reg(reg, results)

    log(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
