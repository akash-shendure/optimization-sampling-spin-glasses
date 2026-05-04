# final smooth-panel figures: sparse random + SK with all four samplers at alpha=2.0
from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.models.sparse_glass import SparseRandomGlass
from spinglass.models.sk import SherringtonKirkpatrick
from spinglass.samplers.metropolis import MetropolisSampler
from spinglass.samplers.langevin import LangevinSampler
from spinglass.samplers.mala import MALASampler
from spinglass.samplers.hmc import HMCSampler
from spinglass.experiments.studies import sampling_beta_sweep
from spinglass.experiments.benchmarks import summarize_sampling_table
from spinglass.experiments.budget import sweeps
from spinglass.plotting.style import set_publication_style

# flush=True so progress is visible when piped to a log file
def log(msg=""):
    print(msg, flush=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("results/final_gaussian")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# alpha=2.0 picked by sweep_alpha_ising as the best mixing default
ALPHA = 2.0
LAM = 0.0  # no non-binary penalty; quartic reg lives in a separate sweep
REG = "linear"
N_CHAINS = 4
TRACE_EVERY = 10  # thin trace storage so D*chains stays manageable
BUDGET = sweeps(200)  # equal compute per sampler so cost comparisons are fair
D = 20  # disorder realizations per beta
N_BETAS = 30

# beta ranges chosen to bracket each model's spin-glass transition
BETAS_SPARSE = np.round(np.linspace(0.25, 5.00, N_BETAS), 4).tolist()
BETAS_SK = np.round(np.linspace(0.15, 3.00, N_BETAS), 4).tolist()

MODELS = {
    "sparse": dict(
        model_class=SparseRandomGlass,
        single_kwargs={"n": [72], "c": [3.0], "disorder": ["gaussian"]},
        n_spins=72,
        betas=BETAS_SPARSE,
        model_name="Sparse Random Glass",
        single_params=r"$n=72$, $c=3$, gaussian",
    ),
    "sk": dict(
        model_class=SherringtonKirkpatrick,
        single_kwargs={"n": [48], "disorder": ["gaussian"]},
        n_spins=48,  # SK is dense so n=48 already costs ~n^2 work
        betas=BETAS_SK,
        model_name="Sherrington-Kirkpatrick",
        single_params=r"$n=48$, gaussian",
    ),
}

# (display name, class, sampling space, sampler kwargs); step sizes hand-tuned
SAMPLERS = [
    ("Metropolis", MetropolisSampler, "discrete", None),
    ("Langevin", LangevinSampler, "relaxed", {"step_size": 0.02}),
    ("MALA", MALASampler, "relaxed", {"step_size": 0.02}),
    ("HMC", HMCSampler, "relaxed", {"step_size": 0.02, "n_leapfrog": 10}),
]

# aggregate per-disorder rows into per-beta mean/std so plots show variability
def disorder_stats(result):
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

# average |m| over the last 75% of each chain's trace, then over chains and disorders
def disorder_magnetization(result):
    by_key = defaultdict(list)
    for record in result["records"]:
        meta = record.get("meta", {})
        trace = record.get("trace") or {}
        if "magnetization" not in trace:
            continue
        beta = meta.get("algorithm_beta")
        did = meta.get("disorder_id", 0)
        m = np.asarray(trace["magnetization"], dtype=np.float64)
        if m.size == 0:
            continue
        keep = max(1, int(0.25 * m.size))  # drop early burn-in tail
        by_key[(beta, did)].append(float(np.mean(np.abs(m[keep:]))))
    by_beta = defaultdict(list)
    for (beta, did), chain_vals in by_key.items():
        by_beta[beta].append(float(np.mean(chain_vals)))
    out = []
    for beta in sorted(by_beta.keys()):
        vals = np.asarray(by_beta[beta], dtype=np.float64)
        out.append({
            "beta": beta,
            "abs_m_mean": float(np.mean(vals)),
            "abs_m_std": float(np.std(vals)) if vals.size > 1 else 0.0,
        })
    return out

# aggregate replica-overlap |q| rows the same way as disorder_stats
def disorder_overlap(result):
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

# run a single (model, sampler) sweep, with on-disk pickle cache for reruns
def run_one(model_name, samp_name, cls, space, skw, cfg):
    cache_path = CACHE_DIR / f"{model_name}_{samp_name}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    t0 = time.time()
    result = sampling_beta_sweep(
        model_class=cfg["model_class"],
        model_kwargs=cfg["single_kwargs"],
        betas=cfg["betas"],
        sampler_class=cls,
        sampler_kwargs=skw,
        space=space,
        n_chains=N_CHAINS, n_steps=2000, burn_in=500,
        trace_every=TRACE_EVERY, n_disorders=D, budget=BUDGET,
        alpha=ALPHA, lam=LAM, reg=REG,
    )
    bundle = {
        "stats": disorder_stats(result),
        "overlap": disorder_overlap(result),
        "magnetization": disorder_magnetization(result),
    }
    with cache_path.open("wb") as f:
        pickle.dump(bundle, f)
    log(f"    [{model_name}] {samp_name}: done in {time.time()-t0:.1f}s")
    return bundle

# attach axis labels, titles, and legends to the 2x4 diagnostics grid
def _dress(axes):
    axes[0, 0].set_xlabel(r"$\beta$"); axes[0, 0].set_ylabel(r"$\langle H \rangle / n$")
    axes[0, 0].set_title("(a) Mean energy per spin"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xlabel(r"$\beta$"); axes[0, 1].set_ylabel("acceptance rate")
    axes[0, 1].set_title("(b) Acceptance rate"); axes[0, 1].set_ylim(-0.02, 1.02)  # rate is a probability
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].set_xlabel(r"$\beta$"); axes[0, 2].set_ylabel("ESS")
    axes[0, 2].set_title("(c) Effective sample size"); axes[0, 2].set_yscale("log")  # spans orders of magnitude
    axes[0, 2].legend(fontsize=8)
    axes[0, 3].set_xlabel(r"$\beta$"); axes[0, 3].set_ylabel(r"$\langle |m| \rangle$")
    axes[0, 3].set_title(r"(d) Mean magnetization $|m|$")
    axes[0, 3].set_ylim(-0.02, 1.02); axes[0, 3].legend(fontsize=8)
    axes[1, 0].set_xlabel(r"$\beta$"); axes[1, 0].set_ylabel(r"$\hat{R}$")
    axes[1, 0].set_title(r"(e) Split-$\hat{R}$"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_xlabel(r"$\beta$"); axes[1, 1].set_ylabel(r"$\hat{\tau}_{\rm int}$")
    axes[1, 1].set_title("(f) Integrated autocorrelation time"); axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].set_xlabel(r"$\beta$"); axes[1, 2].set_ylabel(r"$\langle |q| \rangle$")
    axes[1, 2].set_title(r"(g) Mean replica overlap $|q|$")
    axes[1, 2].set_ylim(-0.05, 1.05); axes[1, 2].legend(fontsize=8)
    axes[1, 3].axis("off")  # 8th slot intentionally blank

# scatter the six diagnostic curves for one sampler onto the shared grid
def _plot_six(axes, stats, overlap, mag, n_spins, color, marker, label, show_eb=True):
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
    if mag:
        bm = np.array([r["beta"] for r in mag])
        mm = np.array([r["abs_m_mean"] for r in mag])
        sm = np.array([r["abs_m_std"] for r in mag]) if show_eb else None
        axes[0, 3].errorbar(bm, mm, yerr=sm, fmt=f"{marker}-", color=color,
                            capsize=capsize, markersize=ms, label=label)
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

# emit both an errorbar and a clean version of the 8-panel figure for one model
def make_figure(model_name, cfg, results):
    palette = plt.cm.tab10(np.arange(len(SAMPLERS)))  # one color per sampler
    title_base = (f"{cfg['model_name']} ({cfg['single_params']}, "
                  rf"$d={D}$, $\alpha={ALPHA}$)")
    markers = ["o", "s", "^", "D"]
    # iterate eb=False then True so the smooth figure is written second
    for eb in [False, True]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        fig.suptitle(title_base, fontsize=14)
        for i, (name, _cls, _sp, _kw) in enumerate(SAMPLERS):
            _plot_six(axes, results[name]["stats"], results[name]["overlap"],
                      results[name].get("magnetization", []),
                      cfg["n_spins"], palette[i], markers[i], name, show_eb=eb)
        _dress(axes)
        suffix = "_eb" if eb else ""
        path = FIG_DIR / f"diagnostics_{model_name}_final{suffix}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        log(f"  saved {path}")
        plt.close(fig)

# top-level driver: run all (model, sampler) cells then build both figures
def main():
    set_publication_style()
    t0 = time.time()
    log(f"Final run: gaussian sparse + SK with all samplers at alpha={ALPHA}")
    log(f"  D={D}, N_BETAS={N_BETAS}, sweeps={BUDGET.value:.0f}, reg={REG}, lam={LAM}")
    log("=" * 60)

    # sweep loop: collect all results before plotting so a plotting bug doesn't kill the long run
    all_results = {}
    for model_name, cfg in MODELS.items():
        log(f"\n[{model_name}]")
        all_results[model_name] = {}
        for samp_name, cls, space, skw in SAMPLERS:
            log(f"  {samp_name} ({space})")
            all_results[model_name][samp_name] = run_one(
                model_name, samp_name, cls, space, skw, cfg)

    # figure assembly after all sweeps complete
    log("\nGenerating figures ...")
    for model_name, cfg in MODELS.items():
        make_figure(model_name, cfg, all_results[model_name])

    log(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
