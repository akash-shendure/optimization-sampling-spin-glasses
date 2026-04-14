# high-level study functions — beta sweeps for sampling/optimization in both spaces
import numpy as np

from ..optimizers.adam import AdamOptimizer
from ..optimizers.simulated_annealing import SimulatedAnnealing
from ..samplers.langevin import LangevinSampler
from ..samplers.metropolis import MetropolisSampler
from .benchmarks import summarize_optimization_table, summarize_sampling_table
from .budget import budget_to_n_steps
from .overlap import summarize_replica_overlaps
from .runner import run_grid

# linear ramp from beta_hot (small) to beta_cold (large) — slow annealing
def linear_beta_schedule(beta_hot, beta_cold, n_steps):
    return np.linspace(float(beta_hot), float(beta_cold), int(n_steps))

def _disorder_seeds(base_seed, n_disorders):
    base = int(base_seed)
    return [base + 1000 * k for k in range(int(n_disorders))]

def _inject_seeds(model_kwargs, n_disorders, base_seed):
    if n_disorders is None or int(n_disorders) <= 1:
        return dict(model_kwargs)
    out = dict(model_kwargs)
    if "seed" in out and isinstance(out["seed"], (list, tuple)) and len(out["seed"]) > 1:
        return out
    out["seed"] = _disorder_seeds(base_seed, n_disorders)
    return out

# build a single concrete model to read n — used for budget resolution
def _probe_n_spins(model_class, model_kwargs):
    first = {}
    for key, value in model_kwargs.items():
        # take first option from list-valued kwargs to materialize one model
        first[key] = value[0] if isinstance(value, (list, tuple)) and len(value) > 0 else value
    if isinstance(first.get("seed"), int):
        pass
    probe = model_class(**first)
    return int(probe.n)

# convert a Budget (or None) to a concrete int n_steps for this n and space
def _resolve_steps(budget, n_steps, n_spins, space):
    if budget is not None:
        return int(budget_to_n_steps(budget, n_spins, space))
    return int(n_steps)

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
    n_disorders=1,
    base_seed=0,
    budget=None,
):
    sampler_kwargs = dict(sampler_kwargs or {})
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    # resolve budget once for the probed n; assumes all replicates share n
    effective_steps = _resolve_steps(budget, n_steps, n_spins, space)
    # scale burn-in proportionally when the budget overrode n_steps
    effective_burn = int(round(burn_in * effective_steps / max(1, int(n_steps))))

    # beta is the primary sweep axis; any user sampler_kwargs ride along as fixed
    algorithm_grid = {"beta": list(map(float, betas))}
    for key, value in sampler_kwargs.items():
        algorithm_grid.setdefault(key, [value])
    algorithm_grid.setdefault("seed", [1000])

    run_kwargs = {
        "n_steps": int(effective_steps),
        "burn_in": int(effective_burn),
        "trace_every": int(trace_every),
    }
    results = run_grid(
        task="sampling",
        space=space,
        model_class=model_class,
        model_grid=model_grid,
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
        "n_disorders": int(max(1, n_disorders or 1)),
        "effective_n_steps": int(effective_steps),
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
    n_disorders=1,
    base_seed=0,
    budget=None,
):
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    # discrete-space budget — single-flip cost model
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "discrete")
    # one schedule per target beta — the sweep axis is the cold endpoint
    algorithm_grid = {
        "beta_schedule": [linear_beta_schedule(beta_hot, b, effective_steps) for b in target_betas],
        "seed": [2000],
    }
    run_kwargs = {"n_steps": int(effective_steps), "trace_every": int(trace_every)}
    if target_energy is not None:
        run_kwargs["target_energy"] = float(target_energy)

    results = run_grid(
        task="optimization",
        space="discrete",
        model_class=model_class,
        model_grid=model_grid,
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
        "n_disorders": int(max(1, n_disorders or 1)),
        "effective_n_steps": int(effective_steps),
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
    n_disorders=1,
    base_seed=0,
    budget=None,
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
        n_disorders=n_disorders,
        base_seed=base_seed,
        budget=budget,
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
    n_disorders=1,
    base_seed=0,
    budget=None,
):
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    # relaxed budget — one step touches all n spins
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "relaxed")
    algorithm_grid = {"lr": list(map(float, lrs)), "seed": [3000]}
    run_kwargs = {
        "n_steps": int(effective_steps),
        "trace_every": int(trace_every),
        "project": True,  # always round the final x back to ±1 for scoring
    }
    results = run_grid(
        task="optimization",
        space="relaxed",
        model_class=model_class,
        model_grid=model_grid,
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
        "n_disorders": int(max(1, n_disorders or 1)),
        "effective_n_steps": int(effective_steps),
    }

# run all four (space, task) panels at once — backs the CLI presets
def canonical_study(
    model_class,
    model_kwargs,
    betas,
    n_steps=2000,
    n_chains=4,
    n_restarts=6,
    n_disorders=1,
    base_seed=0,
    budget=None,
):
    shared = dict(n_disorders=n_disorders, base_seed=base_seed, budget=budget)
    return {
        ("discrete", "sampling"): sampling_beta_sweep(
            model_class, model_kwargs, betas, n_chains=n_chains, n_steps=n_steps, **shared
        ),
        ("discrete", "optimization"): optimization_beta_sweep(
            model_class, model_kwargs, target_betas=betas, n_steps=n_steps, n_restarts=n_restarts, **shared
        ),
        ("relaxed", "sampling"): relaxed_sampling_beta_sweep(
            model_class, model_kwargs, betas, n_chains=n_chains, n_steps=n_steps, **shared
        ),
        # relaxed optimization sweeps lr instead of beta — temperature has no role for adam
        ("relaxed", "optimization"): relaxed_optimization_beta_sweep(
            model_class,
            model_kwargs,
            lrs=[0.02, 0.05, 0.1, 0.2],
            n_steps=n_steps,
            n_restarts=n_restarts,
            **shared,
        ),
    }
