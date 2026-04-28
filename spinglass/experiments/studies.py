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

# geometric beta ramp — spends more steps near the hot end
def geometric_beta_schedule(beta_hot, beta_cold, n_steps):
    return np.geomspace(float(beta_hot), float(beta_cold), int(n_steps))

# geometric in temperature T, returned as beta=1/T (preferred for SA)
def geometric_temperature_schedule(T_hot, T_cold, n_steps):
    T = np.geomspace(float(T_hot), float(T_cold), int(n_steps))
    return 1.0 / T

# linear in T, returned as beta=1/T — gives a 1/(a+bt) shape in beta
def linear_temperature_schedule(T_hot, T_cold, n_steps):
    T = np.linspace(float(T_hot), float(T_cold), int(n_steps))
    return 1.0 / T

# attach a fake "_disorder_id" axis so the grid runner replicates the model
def _inject_replicates(model_kwargs, n_disorders):
    if n_disorders is None or int(n_disorders) <= 1:
        return dict(model_kwargs)
    out = dict(model_kwargs)
    # each value of _disorder_id is a different seed when the model uses rng
    out["_disorder_id"] = list(range(int(n_disorders)))
    return out

# build a single concrete model to read n — used for budget resolution
def _probe_n_spins(model_class, model_kwargs):
    first = {}
    for key, value in model_kwargs.items():
        if key == "_disorder_id":
            continue
        # take first option from list-valued kwargs to materialize one model
        first[key] = value[0] if isinstance(value, (list, tuple)) and len(value) > 0 else value
    probe = model_class(**first)
    return int(probe.n)

# convert a Budget (or None) to a concrete int n_steps for this n and space
def _resolve_steps(budget, n_steps, n_spins, space):
    if budget is not None:
        return int(budget_to_n_steps(budget, n_spins, space))
    return int(n_steps)

# build group-by keys: include any model kwarg that varies + the primary axis
def _grouping_keys(model_kwargs, primary_key):
    keys = []
    for name, value in model_kwargs.items():
        if name == "_disorder_id":
            continue
        # only include axes that actually sweep (len>1); fixed kwargs are uniform
        if isinstance(value, (list, tuple)) and len(value) > 1:
            keys.append(f"model_{name}")
    keys.append(primary_key)
    return keys

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
    alpha=2.0,
    lam=0.0,
    reg="linear",
    n_disorders=1,
    budget=None,
):
    sampler_kwargs = dict(sampler_kwargs or {})
    model_grid = _inject_replicates(model_kwargs, n_disorders)
    n_spins = _probe_n_spins(model_class, model_grid)
    # resolve budget once for the probed n; assumes all replicates share n
    effective_steps = _resolve_steps(budget, n_steps, n_spins, space)
    # scale burn-in proportionally when the budget overrode n_steps
    effective_burn = int(round(burn_in * effective_steps / max(1, int(n_steps))))

    # beta is the primary sweep axis; any user sampler_kwargs ride along as fixed
    algorithm_grid = {"beta": list(map(float, betas))}
    for key, value in sampler_kwargs.items():
        algorithm_grid.setdefault(key, [value])

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
        reg=reg,
        n_chains=int(n_chains),
        keep_trace=True,
        keep_artifacts=bool(keep_artifacts),
        experiment_name=experiment_name or "sampling_beta_sweep",
    )
    # group by the varying model axes plus the algorithm beta column
    group_keys = _grouping_keys(model_grid, "algorithm_beta")
    grouped = summarize_sampling_table(
        results["table"], records=results["records"], group_by=group_keys
    )
    overlaps = summarize_replica_overlaps(results["records"], group_by=group_keys)
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
    budget=None,
):
    model_grid = _inject_replicates(model_kwargs, n_disorders)
    n_spins = _probe_n_spins(model_class, model_grid)
    # discrete-space budget — single-flip cost model
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "discrete")
    # one schedule per target beta — the sweep axis is the cold endpoint
    algorithm_grid = {
        "beta_schedule": [linear_beta_schedule(beta_hot, b, effective_steps) for b in target_betas],
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

    group_keys = _grouping_keys(model_grid, "target_beta")
    grouped = summarize_optimization_table(results["table"], group_by=group_keys)
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
    alpha=2.0,
    lam=0.0,
    reg="linear",
    n_chains=4,
    n_steps=2000,
    burn_in=500,
    trace_every=10,
    experiment_name=None,
    n_disorders=1,
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
        reg=reg,
        n_disorders=n_disorders,
        budget=budget,
    )

# relaxed-space optimization — adam on the surrogate, project at end
def relaxed_optimization_beta_sweep(
    model_class,
    model_kwargs,
    lrs=(0.05,),
    alpha=2.0,
    lam=0.0,
    n_steps=2000,
    n_restarts=6,
    trace_every=25,
    experiment_name=None,
    n_disorders=1,
    budget=None,
):
    model_grid = _inject_replicates(model_kwargs, n_disorders)
    n_spins = _probe_n_spins(model_class, model_grid)
    # relaxed budget — one step touches all n spins
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "relaxed")
    algorithm_grid = {"lr": list(map(float, lrs))}
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
    group_keys = _grouping_keys(model_grid, "algorithm_lr")
    grouped = summarize_optimization_table(results["table"], group_by=group_keys)
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
    budget=None,
):
    # shared kwargs passed to every panel so disorder/budget stay consistent
    shared = dict(n_disorders=n_disorders, budget=budget)
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
