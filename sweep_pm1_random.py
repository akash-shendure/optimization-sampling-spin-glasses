# pm1-coupling rerun: discrete vs relaxed diagnostics plus a size-scaling figure for sparse + SK
from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.models.sparse_glass import SparseRandomGlass
from spinglass.models.sk import SherringtonKirkpatrick
from spinglass.experiments.studies import sampling_beta_sweep, relaxed_sampling_beta_sweep
from spinglass.experiments.benchmarks import summarize_sampling_table
from spinglass.experiments.budget import sweeps
from spinglass.plotting.style import set_publication_style, PANEL_COLORS

# flush=True so progress is visible when piped to a log file
def log(msg=""):
    print(msg, flush=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("results/pm1_random")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_CHAINS = 4
TRACE_EVERY = 10
BUDGET = sweeps(200)  # fixed compute per cell so cost comparisons are fair
D = 15
N_BETAS = 30

# beta ranges chosen to bracket each model's spin-glass transition
BETAS_SPARSE = np.round(np.linspace(0.25, 5.00, N_BETAS), 4).tolist()
BETAS_SK = np.round(np.linspace(0.15, 3.00, N_BETAS), 4).tolist()

MODELS = {
    "sparse": dict(
        model_class=SparseRandomGlass,
        single_kwargs={"n": [72], "c": [3.0], "disorder": ["pm1"]},
        size_key="n",
        sizes=[32, 48, 72, 96, 128],  # geometric-ish ladder for scaling plot
        extra_kwargs={"c": [3.0], "disorder": ["pm1"]},
        betas=BETAS_SPARSE,
        model_name="Sparse Random Glass",
        single_params=r"$n=72$, $c=3$",
    ),
    "sk": dict(
        model_class=SherringtonKirkpatrick,
        single_kwargs={"n": [48], "disorder": ["pm1"]},
        size_key="n",
        sizes=[24, 32, 48, 64, 80],  # SK is dense; smaller top size keeps cost bounded
        extra_kwargs={"disorder": ["pm1"]},
        betas=BETAS_SK,
        model_name="Sherrington-Kirkpatrick",
        single_params=r"$n=48$",
    ),
}

# aggregate per-disorder rows into per-beta mean/std for the 6 diagnostic metrics
def compute_disorder_stats(result):
    per_d = summarize_sampling_table(
        result["table"], records=result["records"],
        group_by=["algorithm_beta", "disorder_id"]
    )
    by_beta = defaultdict(list)
    for row in per_d:
        by_beta[row["algorithm_beta"]].append(row)
    metrics = ["mean_energy", "mean_acceptance_rate",
               "energy_ess", "energy_rhat", "energy_tau_int"]
    out = []
    for beta in sorted(by_beta.keys()):
        rows = by_beta[beta]
        agg = {"beta": beta, "n_disorders": len(rows)}
        for k in metrics:
            vals = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if valid.size else np.nan
            agg[f"{k}_std"] = float(np.nanstd(vals)) if valid.size > 1 else 0.0
        out.append(agg)
    return out

# aggregate replica-overlap |q| rows the same way as compute_disorder_stats
def compute_disorder_overlap_stats(result):
    rows = result.get("overlap", [])
    if not rows:
        return []
    by_beta = defaultdict(list)
    for row in rows:
        by_beta[row["algorithm_beta"]].append(row)
    out = []
    for beta in sorted(by_beta.keys()):
        rs = by_beta[beta]
        abs_qs = np.array([r.get("mean_abs_q", np.nan) for r in rs], dtype=np.float64)
        valid = abs_qs[~np.isnan(abs_qs)]
        out.append({
            "beta": beta,
            "mean_abs_q_mean": float(np.nanmean(abs_qs)) if valid.size else np.nan,
            "mean_abs_q_std": float(np.nanstd(abs_qs)) if valid.size > 1 else 0.0,
        })
    return out

# pickle-cache wrapper: load if path exists else compute fn(*args, **kwargs) and store
def _cached(cache_path, fn, *args, **kwargs):
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    res = fn(*args, **kwargs)
    with cache_path.open("wb") as f:
        pickle.dump(res, f)
    return res

# run a discrete (Metropolis) beta sweep with the shared budget settings
def _run_discrete(model_class, model_kwargs, betas, d):
    return sampling_beta_sweep(
        model_class=model_class, model_kwargs=model_kwargs,
        betas=betas, n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=d, budget=BUDGET,
    )

# run a relaxed-space beta sweep with the shared budget settings
def _run_relaxed(model_class, model_kwargs, betas, d):
    return relaxed_sampling_beta_sweep(
        model_class=model_class, model_kwargs=model_kwargs,
        betas=betas, n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=d, budget=BUDGET,
    )

# bundle stats+overlap so figure code only depends on this compact dict shape
def _summary_bundle(result):
    return {
        "stats": compute_disorder_stats(result),
        "overlap": compute_disorder_overlap_stats(result),
    }

# attach axis labels, titles, and legends to the 2x3 diagnostics grid
def _dress(axes):
    axes[0, 0].set_xlabel(r"$\beta$"); axes[0, 0].set_ylabel(r"$\langle H \rangle / n$")
    axes[0, 0].set_title("(a) Mean energy per spin"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xlabel(r"$\beta$"); axes[0, 1].set_ylabel("acceptance rate")
    axes[0, 1].set_title("(b) Acceptance rate"); axes[0, 1].set_ylim(-0.02, 1.02)  # rate is a probability
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].set_xlabel(r"$\beta$"); axes[0, 2].set_ylabel("ESS")
    axes[0, 2].set_title("(c) Effective sample size"); axes[0, 2].set_yscale("log")
    axes[0, 2].legend(fontsize=8)
    axes[1, 0].set_xlabel(r"$\beta$"); axes[1, 0].set_ylabel(r"$\hat{R}$")
    axes[1, 0].set_title(r"(d) Split-$\hat{R}$"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_xlabel(r"$\beta$"); axes[1, 1].set_ylabel(r"$\hat{\tau}_{\rm int}$")
    axes[1, 1].set_title("(e) Integrated autocorrelation time"); axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].set_xlabel(r"$\beta$"); axes[1, 2].set_ylabel(r"$\langle |q| \rangle$")
    axes[1, 2].set_title(r"(f) Mean replica overlap $|q|$")
    axes[1, 2].set_ylim(-0.05, 1.05); axes[1, 2].legend(fontsize=8)

# scatter the six diagnostic curves for one (sampler, size) cell onto the shared grid
def _plot_six(axes, stats, overlap, n_spins, color, marker, label, show_eb=True):
    ms = 4
    capsize = 2.5 if show_eb else 0
    betas = np.array([r["beta"] for r in stats])

    # shorthand pullers for mean / std arrays keyed by metric name
    def _v(k):
        return np.array([r[f"{k}_mean"] for r in stats])

    def _e(k):
        return np.array([r[f"{k}_std"] for r in stats]) if show_eb else None

    axes[0, 0].errorbar(betas, _v("mean_energy") / n_spins,
                        yerr=_e("mean_energy") / n_spins if show_eb else None,
                        fmt=f"{marker}-", color=color, capsize=capsize, markersize=ms, label=label)
    m = _v("mean_acceptance_rate")
    # gibbs/heat-bath has no MH step so acceptance is undefined; skip if all NaN
    if not np.all(np.isnan(m)):
        axes[0, 1].errorbar(betas, m, yerr=_e("mean_acceptance_rate"),
                            fmt=f"{marker}-", color=color, capsize=capsize, markersize=ms, label=label)
    axes[0, 2].errorbar(betas, _v("energy_ess"), yerr=_e("energy_ess"),
                        fmt=f"{marker}-", color=color, capsize=capsize, markersize=ms, label=label)
    axes[1, 0].errorbar(betas, _v("energy_rhat"), yerr=_e("energy_rhat"),
                        fmt=f"{marker}-", color=color, capsize=capsize, markersize=ms, label=label)
    axes[1, 1].errorbar(betas, _v("energy_tau_int"), yerr=_e("energy_tau_int"),
                        fmt=f"{marker}-", color=color, capsize=capsize, markersize=ms, label=label)
    if overlap:
        bq = np.array([r["beta"] for r in overlap])
        mq = np.array([r["mean_abs_q_mean"] for r in overlap])
        sq = np.array([r["mean_abs_q_std"] for r in overlap]) if show_eb else None
        axes[1, 2].errorbar(bq, mq, yerr=sq, fmt=f"{marker}-", color=color,
                            capsize=capsize, markersize=ms, label=label)

# perceptually ordered color ramp for size-scaling lines
def _size_colors(n):
    return [plt.cm.plasma(x) for x in np.linspace(0.12, 0.82, n)]

# emit both errorbar and clean diagnostics figures contrasting discrete vs relaxed
def plot_diagnostics(name, disc_b, relax_b, cfg):
    # sparse/SK store n directly; ising2d stores L so n = L^2
    if "L" in cfg["single_kwargs"]:
        n_sp = cfg["single_kwargs"]["L"][0] ** 2
    else:
        n_sp = cfg["single_kwargs"]["n"][0]
    title = f"{cfg['model_name']} (pm1, {cfg['single_params']}, $d={D}$)"
    for eb in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=14)
        _plot_six(axes, disc_b["stats"], disc_b["overlap"], n_sp,
                  PANEL_COLORS["discrete_sampling"], "o", "Metropolis", show_eb=eb)
        _plot_six(axes, relax_b["stats"], relax_b["overlap"], n_sp,
                  PANEL_COLORS["relaxed_sampling"], "s", "Langevin", show_eb=eb)
        _dress(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"diagnostics_{name}_pm1{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)

# emit both errorbar and clean size-scaling figures for the discrete sampler
def plot_scaling(name, scaling_data, cfg):
    sizes = cfg["sizes"]
    colors = _size_colors(len(sizes))
    all_stats = []
    for sz in sizes:
        data = scaling_data[sz]
        all_stats.append((sz, data["n_spins"], data["disc"]["stats"], data["disc"]["overlap"]))
    title = f"{cfg['model_name']} (pm1, $d={D}$)"
    for eb in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=14)
        for i, (sz, n_sp, stats, ov) in enumerate(all_stats):
            _plot_six(axes, stats, ov, n_sp, colors[i], "o", f"$n={n_sp}$", show_eb=eb)
        _dress(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"scaling_{name}_pm1{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)

# top-level driver: run single-size diagnostics, then a 5-size scaling sweep, then plot
def main():
    set_publication_style()
    t_global = time.time()
    log("Pm1 disorder re-run for sparse + SK")
    log(f"  D={D}, N_BETAS={N_BETAS}, sweeps={BUDGET.value:.0f}")
    log("=" * 60)

    # single-size diagnostics: discrete vs relaxed at the reference n
    log("\n--- Diagnostics (single size, discrete vs relaxed) ---")
    single = {}
    for name, cfg in MODELS.items():
        log(f"\n  [{name}]")
        mc, mk = cfg["model_class"], cfg["single_kwargs"]
        t0 = time.time()
        disc = _cached(CACHE_DIR / f"{name}_pm1_disc_single.pkl", _run_discrete, mc, mk, cfg["betas"], D)
        log(f"    Metropolis done in {time.time()-t0:.1f}s")
        t0 = time.time()
        relax = _cached(CACHE_DIR / f"{name}_pm1_relax_single.pkl", _run_relaxed, mc, mk, cfg["betas"], D)
        log(f"    Langevin done in {time.time()-t0:.1f}s")
        single[name] = (_summary_bundle(disc), _summary_bundle(relax))

    log("\n  Plotting diagnostic figures ...")
    for name, cfg in MODELS.items():
        plot_diagnostics(name, single[name][0], single[name][1], cfg)

    # size-scaling sweep: discrete sampler only (relaxed adds little here and doubles cost)
    log("\n--- Scaling (5 sizes, discrete only) ---")
    scaling = {}
    for name, cfg in MODELS.items():
        sk = cfg["size_key"]
        scaling[name] = {}
        for sz in cfg["sizes"]:
            mk = {sk: [sz]}
            mk.update(cfg["extra_kwargs"])
            tag = f"{sk}={sz}"
            log(f"\n  [{name}] {tag}")
            t0 = time.time()
            disc = _cached(CACHE_DIR / f"{name}_pm1_disc_{sk}{sz}.pkl",
                           _run_discrete, cfg["model_class"], mk, cfg["betas"], D)
            log(f"    done in {time.time()-t0:.1f}s")
            n_sp = sz * sz if sk == "L" else sz  # L means a square lattice
            scaling[name][sz] = {"disc": _summary_bundle(disc), "n_spins": n_sp}

    log("\n  Plotting scaling figures ...")
    for name, cfg in MODELS.items():
        plot_scaling(name, scaling[name], cfg)

    log(f"\nTotal time: {time.time()-t_global:.0f}s")

if __name__ == "__main__":
    main()
