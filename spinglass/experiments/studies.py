"""canonical benchmark study: beta sweep across models and methods.

this wires the existing run_grid / benchmark machinery into the specific
comparisons the proposal asks about — a temperature sweep for a fixed model,
with matched optimizer and sampler pairs, so downstream plotting can draw a
single "difficulty vs beta" curve per method per model.

two knobs matter for making the comparisons statistically and computationally
fair:
  n_disorders — number of independent disorder realizations averaged per cell
                of the grid. without this, glassy regime results are dominated
                by instance-to-instance variation.
  budget      — Budget object from experiments.budget that resolves to a
                fair n_steps given the model size and discrete/relaxed space.
                int n_steps is still accepted for legacy callers."""
import numpy as np

from ..optimizers.adam import AdamOptimizer
from ..optimizers.simulated_annealing import SimulatedAnnealing
from ..samplers.langevin import LangevinSampler
from ..samplers.metropolis import MetropolisSampler
from .benchmarks import summarize_optimization_table, summarize_sampling_table
from .budget import budget_to_n_steps
from .overlap import summarize_replica_overlaps
from .runner import run_grid


def linear_beta_schedule(beta_hot, beta_cold, n_steps):
    """linear ramp from beta_hot (high T) to beta_cold (low T), length n_steps."""
    return np.linspace(float(beta_hot), float(beta_cold), int(n_steps))


def _disorder_seeds(base_seed, n_disorders):
    """deterministic spread of disorder seeds from a single base_seed."""
    base = int(base_seed)
    return [base + 1000 * k for k in range(int(n_disorders))]


def _inject_seeds(model_kwargs, n_disorders, base_seed):
    """expand a model_kwargs grid so every (model, beta) cell averages over
    n_disorders independent disorder realizations."""
    if n_disorders is None or int(n_disorders) <= 1:
        return dict(model_kwargs)
    out = dict(model_kwargs)
    # if caller already passed a list of seeds, respect it
    if "seed" in out and isinstance(out["seed"], (list, tuple)) and len(out["seed"]) > 1:
        return out
    out["seed"] = _disorder_seeds(base_seed, n_disorders)
    return out


def _probe_n_spins(model_class, model_kwargs):
    """build a tiny probe of the first grid cell to read n_spins for budget math."""
    first = {}
    for key, value in model_kwargs.items():
        first[key] = value[0] if isinstance(value, (list, tuple)) and len(value) > 0 else value
    if isinstance(first.get("seed"), int):
        pass
    probe = model_class(**first)
    return int(probe.n)


def _resolve_steps(budget, n_steps, n_spins, space):
    """pick the effective n_steps for a run given Budget / legacy n_steps."""
    if budget is not None:
        return int(budget_to_n_steps(budget, n_spins, space))
    return int(n_steps)


def _grouping_keys(model_kwargs, primary_key):
    """model kwargs with more than one value become part of the grouping key,
    so sweeps across multiple L / n / c / disorder variants never mix records
    from different model sizes into the same grouped row."""
    keys = []
    for name, value in model_kwargs.items():
        if name == "seed":
            continue  # disorder seeds are the thing we AVERAGE over
        if isinstance(value, (list, tuple)) and len(value) > 1:
            keys.append(f"model_{name}")
    keys.append(primary_key)
    return keys


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
    """run `sampler_class` at each target beta, with n_chains independent chains.

    if n_disorders > 1, the grid also fans out over that many independent
    disorder realizations per (model, beta) cell, and grouped summaries average
    across them."""
    sampler_kwargs = dict(sampler_kwargs or {})
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    effective_steps = _resolve_steps(budget, n_steps, n_spins, space)
    effective_burn = int(round(burn_in * effective_steps / max(1, int(n_steps))))

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
    """run SimulatedAnnealing with a linear schedule [beta_hot, target_beta]
    at each target beta. best energy vs target beta is the difficulty curve."""
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "discrete")
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
    # tag each row with the target beta = end of the schedule it used
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
        n_disorders=n_disorders,
        base_seed=base_seed,
        budget=budget,
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
    n_disorders=1,
    base_seed=0,
    budget=None,
):
    """Adam on the relaxed surrogate across learning rates, with projection.

    Adam has no beta knob, so the sweep dimension here is lr; the resulting
    "best projected energy vs lr" curve is the relaxed analog of the
    SA difficulty curve."""
    model_grid = _inject_seeds(model_kwargs, n_disorders, base_seed)
    n_spins = _probe_n_spins(model_class, model_grid)
    effective_steps = _resolve_steps(budget, n_steps, n_spins, "relaxed")
    algorithm_grid = {"lr": list(map(float, lrs)), "seed": [3000]}
    run_kwargs = {
        "n_steps": int(effective_steps),
        "trace_every": int(trace_every),
        "project": True,
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
    """the full four-panel study: discrete + relaxed, optimization + sampling.

    budget applies uniformly to all four panels so comparisons stay fair.
    n_disorders fans each cell over independent disorder realizations."""
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
        ("relaxed", "optimization"): relaxed_optimization_beta_sweep(
            model_class,
            model_kwargs,
            lrs=[0.02, 0.05, 0.1, 0.2],
            n_steps=n_steps,
            n_restarts=n_restarts,
            **shared,
        ),
    }
