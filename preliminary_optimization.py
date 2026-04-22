# early optimization figures: discrete vs relaxed optimizers and SA/PT schedules
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinglass.hamiltonian.discrete import DiscreteHamiltonian
from spinglass.hamiltonian.relaxed import RelaxedHamiltonian
from spinglass.models.ising2d import IsingFerromagnet2D
from spinglass.optimizers.simulated_annealing import SimulatedAnnealing
from spinglass.optimizers.discrete_greedy import GreedySpinDescent
from spinglass.optimizers.gradient_descent import GradientDescentOptimizer
from spinglass.optimizers.adam import AdamOptimizer
from spinglass.optimizers.parallel_tempering import ParallelTemperingOptimizer
from spinglass.experiments.studies import (
    geometric_beta_schedule,
    geometric_temperature_schedule,
    linear_beta_schedule,
)
from spinglass.plotting.style import set_publication_style

# flush=True so progress is visible under nohup/redirect
def log(msg=""):
    print(msg, flush=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

L = 14
MODEL_NAME = "2D Ising Ferromagnet"
GROUND_STATE = -2.0 * L * L  # why: all-aligned ferromagnet, 2 bonds per spin
T_HOT = 10.0
T_COLD = 0.1
BETA_HOT = 1.0 / T_HOT
BETA_COLD = 1.0 / T_COLD

LW = 1.5

D_OPT = 5  # why: 5 trials per optimizer to show run-to-run variability

# (name, class, search space, config, n_steps, trace_every)
OPTIMIZERS = [
    ("Simulated Annealing", SimulatedAnnealing, "discrete",
     {"T_hot": T_HOT, "T_cold": T_COLD}, 200000, 200),
    ("Greedy Descent", GreedySpinDescent, "discrete", {}, 200, 2),
    ("Gradient Descent", GradientDescentOptimizer, "relaxed", {"lr": 0.05}, 200, 2),
    ("Adam", AdamOptimizer, "relaxed", {"lr": 0.05}, 200, 2),
]

# running min over the trace; in relaxed space use projected_energy (discrete H of round(x))
def _running_best(trace, space):
    if space == "discrete":
        return np.asarray(trace["best_energy"], dtype=np.float64)
    pe = np.asarray(trace.get("projected_energy", []), dtype=np.float64)
    out = np.empty_like(pe)
    running = np.inf
    for i, v in enumerate(pe):
        if np.isfinite(v) and v < running:
            running = v
        out[i] = running
    return out

# one trace per (optimizer, trial); dispatches on optimizer name
def _run_optimizer(name, opt_class, space, config, n_steps, trace_every):
    model = IsingFerromagnet2D(L=L)
    discrete_ham = DiscreteHamiltonian(model)
    if space == "discrete":
        ham = discrete_ham
    else:
        # alpha=1, lam=0 here so the surrogate matches the optimizer paper baseline
        ham = RelaxedHamiltonian(model, alpha=1.0, lam=0.0)

    if name == "Simulated Annealing":
        schedule = geometric_temperature_schedule(config["T_hot"], config["T_cold"], n_steps)
        opt = opt_class(ham, schedule)
        run_kwargs = {"n_steps": n_steps, "trace_every": trace_every}
    elif name == "Greedy Descent":
        opt = opt_class(ham)
        run_kwargs = {"n_steps": n_steps, "trace_every": trace_every}
    else:
        opt = opt_class(ham, lr=config["lr"])
        run_kwargs = {
            "n_steps": n_steps, "trace_every": trace_every,
            "project": True, "discrete_hamiltonian": discrete_ham,  # why: track discrete energy of rounded x
        }
    result = opt.run(**run_kwargs)
    return result["trace"]

# collect D_OPT trials, each a (step, running-best) pair
def _aggregate_optimizer(name, opt_class, space, config, n_steps, trace_every):
    trials = []
    for _ in range(D_OPT):
        trace = _run_optimizer(name, opt_class, space, config, n_steps, trace_every)
        step = np.asarray(trace["step"], dtype=np.float64)
        best = _running_best(trace, space)
        trials.append((step, best))
    return trials

# trim flat tails so SA panel doesn't waste x-space after convergence
def _trim_to_convergence(trials, buffer_frac=0.05):
    last_steps = []
    for step, best in trials:
        diffs = np.diff(best)
        neg = np.where(diffs < 0)[0]
        # step after last improvement; fall back to step[0] if never improved
        last_steps.append(float(step[neg[-1] + 1]) if neg.size else float(step[0]))
    cutoff = max(last_steps) * (1.0 + buffer_frac)
    out = []
    for step, best in trials:
        mask = step <= cutoff
        out.append((step[mask], best[mask]))
    return out

# 2x2 figure: one optimizer per panel, sharing y-range with SA for visual comparison
def make_optimizer_figure():
    log(f"Preliminary optimization comparison (Ising, n={L * L})")
    log(f"  d={D_OPT} trials per optimizer")
    log("=" * 60)

    # run each optimizer in turn
    results = []
    palette = plt.cm.tab10(np.arange(len(OPTIMIZERS)))
    color_map = {}
    for idx, (name, cls, space, cfg, n_steps, te) in enumerate(OPTIMIZERS):
        color_map[name] = palette[idx]
        log(f"\n  [{name}]  n_steps={n_steps}")
        t_s = time.time()
        trials = _aggregate_optimizer(name, cls, space, cfg, n_steps, te)
        if name == "Simulated Annealing":
            trials = _trim_to_convergence(trials)
        lens = [len(s) for s, _ in trials]
        log(f"    done in {time.time() - t_s:.1f}s  (trace lengths: {lens})")
        results.append((name, trials, space))

    # figure assembly
    log("\n  Generating figure ...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.625), constrained_layout=True)
    fig.suptitle(f"{MODEL_NAME}  ($n={L * L}$)", fontsize=14)
    gs_line = None
    for idx, (ax, (name, trials, _space)) in enumerate(zip(axes.ravel(), results)):
        line = ax.axhline(
            GROUND_STATE, linestyle="--", color="black", linewidth=LW,
            zorder=1, label="ground state",
        )
        if idx == 0:
            gs_line = line
        for step, best in trials:
            ax.plot(step, best, "-", color=color_map[name], linewidth=LW, zorder=2)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(r"best $H$")
        ax.tick_params(labelsize=11.5)
    axes.ravel()[0].legend(handles=[gs_line], loc="upper right", frameon=True)

    # use SA's y-range everywhere; SA covers the worst-to-best span
    sa_idx = next(i for i, (name, *_) in enumerate(results) if name == "Simulated Annealing")
    sa_ylim = axes.ravel()[sa_idx].get_ylim()
    for ax in axes.ravel():
        ax.set_ylim(sa_ylim)

    # report SA's max trial-end step as x-cutoff for the schedule figure
    sa_trials = results[sa_idx][1]
    sa_cutoff = max(float(step[-1]) for step, _ in sa_trials)

    path = FIG_DIR / "preliminary_optimization.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    log(f"  saved {path}")
    plt.close(fig)
    return sa_cutoff

D_SCHED = 5
K_PT = 5            # why: 5-rung beta ladder for parallel tempering
N_STEPS_SCHED = 200_000
N_TRACE_SCHED = 1000
SWAP_INTERVAL = L * L  # why: attempt swap every full sweep

SCHEDULE_METHODS = [
    (r"Simulated Annealing (linear $\beta$)", "sa", "linear"),
    (r"Simulated Annealing (geometric $\beta$)", "sa", "geometric"),
    (r"Parallel Tempering (linear $\beta$ ladder)", "pt", "linear"),
    (r"Parallel Tempering (geometric $\beta$ ladder)", "pt", "geometric"),
]

# one SA run with the given beta schedule kind
def _run_sa_schedule(kind):
    model = IsingFerromagnet2D(L=L)
    ham = DiscreteHamiltonian(model)
    if kind == "linear":
        schedule = linear_beta_schedule(BETA_HOT, BETA_COLD, N_STEPS_SCHED)
    else:
        schedule = geometric_beta_schedule(BETA_HOT, BETA_COLD, N_STEPS_SCHED)
    opt = SimulatedAnnealing(ham, schedule)
    trace_every = max(1, N_STEPS_SCHED // N_TRACE_SCHED)
    result = opt.run(n_steps=N_STEPS_SCHED, trace_every=trace_every)
    step = np.asarray(result["trace"]["step"], dtype=np.float64)
    best = np.asarray(result["trace"]["best_energy"], dtype=np.float64)
    return step, best

# one PT run with linear or geometric beta ladder
def _run_pt_schedule(kind):
    model = IsingFerromagnet2D(L=L)
    ham = DiscreteHamiltonian(model)
    if kind == "linear":
        betas = np.linspace(BETA_HOT, BETA_COLD, K_PT)
    else:
        betas = np.geomspace(BETA_HOT, BETA_COLD, K_PT)
    trace_every = max(1, N_STEPS_SCHED // N_TRACE_SCHED)
    opt = ParallelTemperingOptimizer(ham, betas=betas, swap_interval=SWAP_INTERVAL)
    result = opt.run(n_steps=N_STEPS_SCHED, trace_every=trace_every)
    step = np.asarray(result["trace"]["step"], dtype=np.float64)
    best = np.asarray(result["trace"]["best_energy"], dtype=np.float64)
    return step, best

# D_SCHED trials per schedule method
def _aggregate_schedule(method, kind):
    trials = []
    for _ in range(D_SCHED):
        if method == "sa":
            trials.append(_run_sa_schedule(kind))
        else:
            trials.append(_run_pt_schedule(kind))
    return trials

# 2x2 schedule figure; x is clipped to SA-figure cutoff for a fair comparison
def make_schedule_figure(x_cutoff):
    log(f"\nSchedule comparison (Ising, n={L * L})")
    log(f"  n_steps={N_STEPS_SCHED}, d={D_SCHED}, beta_hot={BETA_HOT}, beta_cold={BETA_COLD}")
    log(f"  PT: k={K_PT}, swap_interval={SWAP_INTERVAL}")
    log("=" * 60)

    # run each schedule
    results = []
    palette = plt.cm.tab10(np.arange(len(SCHEDULE_METHODS)))
    for idx, (name, method, kind) in enumerate(SCHEDULE_METHODS):
        log(f"\n  [{name}]")
        t_s = time.time()
        trials = _aggregate_schedule(method, kind)
        log(f"    done in {time.time() - t_s:.1f}s")
        results.append((name, trials, palette[idx]))

    # figure assembly
    log("\n  Generating figure ...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.625), constrained_layout=True)
    fig.suptitle(f"{MODEL_NAME}  ($n={L * L}$)", fontsize=14)
    gs_line = None
    for idx, (ax, (name, trials, color)) in enumerate(zip(axes.ravel(), results)):
        line = ax.axhline(
            GROUND_STATE, linestyle="--", color="black", linewidth=LW,
            zorder=1, label="ground state",
        )
        if idx == 0:
            gs_line = line
        for step, best in trials:
            ax.plot(step, best, "-", color=color, linewidth=LW, zorder=2)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(r"best $H$")
        ax.tick_params(labelsize=11.5)
    axes.ravel()[0].legend(handles=[gs_line], loc="upper right", frameon=True)

    # share geometric-SA y-range and SA-figure x-cutoff across all panels
    sa_ylim = axes.ravel()[1].get_ylim()
    for ax in axes.ravel():
        ax.set_ylim(sa_ylim)
        ax.set_xlim(0, x_cutoff)

    path = FIG_DIR / "preliminary_optimization_schedule.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    log(f"  saved {path}")
    plt.close(fig)

# top-level driver: optimizer-comparison figure -> schedule-comparison figure
def main():
    set_publication_style()
    t0 = time.time()
    sa_cutoff = make_optimizer_figure()
    make_schedule_figure(sa_cutoff)
    log(f"\nTotal time: {time.time() - t0:.0f}s")
    log("Done.")

if __name__ == "__main__":
    main()
