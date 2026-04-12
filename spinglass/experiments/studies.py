"""canonical benchmark study: beta sweep across models and methods.

this wires the existing run_grid / benchmark machinery into the specific
comparisons the proposal asks about — a temperature sweep for a fixed model,
with matched optimizer and sampler pairs, so downstream plotting can draw a
single "difficulty vs beta" curve per method per model."""
import numpy as np

from ..optimizers.adam import AdamOptimizer
from ..optimizers.simulated_annealing import SimulatedAnnealing
from ..samplers.langevin import LangevinSampler
from ..samplers.metropolis import MetropolisSampler
from .benchmarks import summarize_optimization_table, summarize_sampling_table
from .overlap import summarize_replica_overlaps
from .runner import run_grid


def linear_beta_schedule(beta_hot, beta_cold, n_steps):
    """linear ramp from beta_hot (high T) to beta_cold (low T), length n_steps."""
    return np.linspace(float(beta_hot), float(beta_cold), int(n_steps))


def sampling_beta_sweep(
    model_class,
    model_kwargs,
    betas,
    sampler_class=MetropolisSampler,
    sampler_kwargs=None,
    n_chains=4,
    n_steps=2000,
    burn_in=500,
    trace_every=10,
    experiment_name=None,
    keep_artifacts=True,
    space="discrete",
    alpha=1.0,
    lam=0.0,
):
    """run `sampler_class` at each target beta, with n_chains independent chains.

    returns a dict carrying records, a flat table, grouped summaries, and a
    replica-overlap summary per beta. keep_artifacts is on by default so
    downstream overlap helpers have final states available."""
    sampler_kwargs = dict(sampler_kwargs or {})
    algorithm_grid = {"beta": list(map(float, betas))}
    for key, value in sampler_kwargs.items():
        algorithm_grid.setdefault(key, [value])
    algorithm_grid.setdefault("seed", [1000])

    run_kwargs = {
        "n_steps": int(n_steps),
        "burn_in": int(burn_in),
        "trace_every": int(trace_every),
    }
    results = run_grid(
        task="sampling",
        space=space,
        model_class=model_class,
        model_grid=model_kwargs,
        algorithm_class=sampler_class,
        algorithm_grid=algorithm_grid,
        run_kwargs=run_kwargs,
        alpha=alpha,
        lam=lam,
        n_chains=int(n_chains),
        keep_trace=True,
        keep_artifacts=bool(keep_artifacts),
        experiment_name=experiment_name or "sampling_beta_sweep",
    )
    grouped = summarize_sampling_table(
        results["table"], records=results["records"], group_by=["algorithm_beta"]
    )
    overlaps = summarize_replica_overlaps(results["records"], group_by=["algorithm_beta"])
    return {
        "records": results["records"],
        "table": results["table"],
        "grouped": grouped,
        "overlap": overlaps,
    }


def optimization_beta_sweep(
    model_class,
    model_kwargs,
    target_betas,
    optimizer_class=SimulatedAnnealing,
    beta_hot=0.05,
    n_steps=2000,
    n_restarts=6,
    trace_every=25,
    target_energy=None,
    experiment_name=None,
    keep_artifacts=False,
):
    """run SimulatedAnnealing with a linear schedule [beta_hot, target_beta]
    at each target beta. best energy vs target beta is the difficulty curve."""
    algorithm_grid = {
        "beta_schedule": [linear_beta_schedule(beta_hot, b, n_steps) for b in target_betas],
        "seed": [2000],
    }
    run_kwargs = {"n_steps": int(n_steps), "trace_every": int(trace_every)}
    if target_energy is not None:
        run_kwargs["target_energy"] = float(target_energy)

    results = run_grid(
        task="optimization",
        space="discrete",
        model_class=model_class,
        model_grid=model_kwargs,
        algorithm_class=optimizer_class,
        algorithm_grid=algorithm_grid,
        run_kwargs=run_kwargs,
        n_restarts=int(n_restarts),
        keep_trace=True,
        keep_artifacts=bool(keep_artifacts),
        experiment_name=experiment_name or "optimization_beta_sweep",
    )
    # tag each row with the target beta = end of the schedule it used
    for row in results["table"]:
        schedule = row.get("algorithm_beta_schedule")
        if schedule is not None:
            row["target_beta"] = float(np.asarray(schedule)[-1])
    for record in results["records"]:
        schedule = record["meta"].get("algorithm_beta_schedule")
        if schedule is not None:
            record["meta"]["target_beta"] = float(np.asarray(schedule)[-1])

    grouped = summarize_optimization_table(results["table"], group_by=["target_beta"])
    return {
        "records": results["records"],
        "table": results["table"],
        "grouped": grouped,
    }


def relaxed_sampling_beta_sweep(
    model_class,
    model_kwargs,
    betas,
    step_size=0.02,
    alpha=1.0,
    lam=0.0,
    n_chains=4,
    n_steps=2000,
    burn_in=500,
    trace_every=10,
    experiment_name=None,
):
    """Langevin beta sweep on the relaxed surrogate — relaxed counterpart to
    sampling_beta_sweep. uses project=True so overlap helpers can operate on
    the stored projected_state artifacts."""
    return sampling_beta_sweep(
        model_class=model_class,
        model_kwargs=model_kwargs,
        betas=betas,
        sampler_class=LangevinSampler,
        sampler_kwargs={"step_size": step_size},
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        trace_every=trace_every,
        experiment_name=experiment_name or "relaxed_sampling_beta_sweep",
        keep_artifacts=True,
        space="relaxed",
        alpha=alpha,
        lam=lam,
    )


def relaxed_optimization_beta_sweep(
    model_class,
    model_kwargs,
    lrs=(0.05,),
    alpha=1.0,
    lam=0.0,
    n_steps=2000,
    n_restarts=6,
    trace_every=25,
    experiment_name=None,
):
    """Adam on the relaxed surrogate across learning rates, with projection.

    Adam has no beta knob, so the sweep dimension here is lr; the resulting
    "best projected energy vs lr" curve is the relaxed analog of the
    SA difficulty curve."""
    algorithm_grid = {"lr": list(map(float, lrs)), "seed": [3000]}
    run_kwargs = {
        "n_steps": int(n_steps),
        "trace_every": int(trace_every),
        "project": True,
    }
    results = run_grid(
        task="optimization",
        space="relaxed",
        model_class=model_class,
        model_grid=model_kwargs,
        algorithm_class=AdamOptimizer,
        algorithm_grid=algorithm_grid,
        run_kwargs=run_kwargs,
        alpha=alpha,
        lam=lam,
        n_restarts=int(n_restarts),
        keep_trace=True,
        keep_artifacts=False,
        experiment_name=experiment_name or "relaxed_optimization_beta_sweep",
    )
    grouped = summarize_optimization_table(results["table"], group_by=["algorithm_lr"])
    return {
        "records": results["records"],
        "table": results["table"],
        "grouped": grouped,
    }


def canonical_study(model_class, model_kwargs, betas, n_steps=2000, n_chains=4, n_restarts=6):
    """the full four-panel study: discrete + relaxed, optimization + sampling.

    this is the single entry point the proposal's "first benchmark study"
    asks for. returns a nested dict keyed by (space, task)."""
    return {
        ("discrete", "sampling"): sampling_beta_sweep(
            model_class, model_kwargs, betas, n_chains=n_chains, n_steps=n_steps
        ),
        ("discrete", "optimization"): optimization_beta_sweep(
            model_class, model_kwargs, target_betas=betas, n_steps=n_steps, n_restarts=n_restarts
        ),
        ("relaxed", "sampling"): relaxed_sampling_beta_sweep(
            model_class, model_kwargs, betas, n_chains=n_chains, n_steps=n_steps
        ),
        ("relaxed", "optimization"): relaxed_optimization_beta_sweep(
            model_class,
            model_kwargs,
            lrs=[0.02, 0.05, 0.1, 0.2],
            n_steps=n_steps,
            n_restarts=n_restarts,
        ),
    }
