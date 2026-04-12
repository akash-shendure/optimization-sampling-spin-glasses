# high-level study functions — beta sweeps for sampling/optimization in both spaces
import numpy as np

from ..optimizers.adam import AdamOptimizer
from ..optimizers.simulated_annealing import SimulatedAnnealing
from ..samplers.langevin import LangevinSampler
from ..samplers.metropolis import MetropolisSampler
from .benchmarks import summarize_optimization_table, summarize_sampling_table
from .overlap import summarize_replica_overlaps
from .runner import run_grid

# linear ramp from beta_hot (small) to beta_cold (large) — slow annealing
def linear_beta_schedule(beta_hot, beta_cold, n_steps):
    return np.linspace(float(beta_hot), float(beta_cold), int(n_steps))

# discrete-space sampling sweep over beta — returns records + summary + overlaps
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
    sampler_kwargs = dict(sampler_kwargs or {})
    # beta is the primary sweep axis; any user sampler_kwargs ride along as fixed
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

# simulated-annealing sweep — target_betas define the cold endpoint of each ramp
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
    # one schedule per target beta — the sweep axis is the cold endpoint
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
    # post-process: stamp target_beta (the cold endpoint) onto every row/record
    # so grouping uses a scalar key instead of the full array
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

# relaxed-space sampling — drives the smooth surrogate with langevin/ula
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
    # thin wrapper — delegates to discrete sampler with space="relaxed" + langevin
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

# relaxed-space optimization — adam on the surrogate, project at end
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
    algorithm_grid = {"lr": list(map(float, lrs)), "seed": [3000]}
    run_kwargs = {
        "n_steps": int(n_steps),
        "trace_every": int(trace_every),
        "project": True,  # always round the final x back to ±1 for scoring
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
        # relaxed optimization sweeps lr instead of beta — temperature has no role for adam
        ("relaxed", "optimization"): relaxed_optimization_beta_sweep(
            model_class,
            model_kwargs,
            lrs=[0.02, 0.05, 0.1, 0.2],
            n_steps=n_steps,
            n_restarts=n_restarts,
        ),
    }
