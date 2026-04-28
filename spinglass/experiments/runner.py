# core sweep runner — run_single executes one config, run_grid iterates the cartesian product
from copy import deepcopy

from .builders import build_algorithm, build_hamiltonian, build_model
from .grids import merge_dicts, parameter_grid
from ..hamiltonian.discrete import DiscreteHamiltonian

# execute one (model, hamiltonian, algorithm) combo and return a record dict
# the chain/restart/disorder ids are stamped into meta for later grouping
def run_single(
    task,
    space,
    model_class,
    model_kwargs,
    algorithm_class,
    algorithm_kwargs,
    run_kwargs,
    alpha=2.0,
    lam=0.0,
    reg="linear",
    keep_trace=True,
    keep_artifacts=False,
    experiment_name=None,
    chain_id=None,
    restart_id=None,
    disorder_id=None,
):
    # deep-copy so the caller's dicts aren't mutated by builders/algorithms
    model_kwargs = deepcopy(model_kwargs)
    algorithm_kwargs = deepcopy(algorithm_kwargs)
    run_kwargs = deepcopy(run_kwargs)

    # _disorder_id is a grid-only marker — strip before constructing the model
    model_kwargs.pop("_disorder_id", None)

    model = build_model(model_class, **model_kwargs)
    hamiltonian = build_hamiltonian(model, space=space, alpha=alpha, lam=lam, reg=reg)
    algorithm = build_algorithm(algorithm_class, hamiltonian, **algorithm_kwargs)

    # relaxed runs should project back to ±1 unless caller explicitly opts out
    if space == "relaxed" and "project" not in run_kwargs:
        run_kwargs["project"] = True
    # projection needs the discrete hamiltonian to score the rounded state
    if space == "relaxed" and run_kwargs.get("project", False):
        run_kwargs.setdefault("discrete_hamiltonian", DiscreteHamiltonian(model))

    result = algorithm.run(**run_kwargs)

    # meta — flat dict used for grouping and table rows
    meta = {
        "experiment_name": experiment_name,
        "task": task,
        "space": space,
        "model_name": model.__class__.__name__,
        "algorithm_name": algorithm_class.__name__,
        "chain_id": chain_id,
        "restart_id": restart_id,
        "disorder_id": disorder_id,
        # alpha/lam/reg only meaningful in relaxed space — None elsewhere
        "alpha": float(alpha) if space == "relaxed" else None,
        "lam": float(lam) if space == "relaxed" else None,
        "reg": str(reg) if space == "relaxed" else None,
    }
    # prefix model/algorithm/run kwargs so columns don't clash
    meta.update(_flatten_prefixed("model", model.describe()))
    meta.update(_flatten_prefixed("algorithm", algorithm_kwargs))
    meta.update(_flatten_prefixed("run", _clean_run_kwargs(run_kwargs)))

    return {
        "meta": meta,
        "summary": result["summary"],
        # traces / artifacts can be huge — opt-in to keep memory bounded
        "trace": result["trace"] if keep_trace else None,
        "artifacts": result["artifacts"] if keep_artifacts else None,
    }

# nested {meta, summary} -> flat row suitable for tabular analysis
def flatten_record(record):
    return merge_dicts(record["meta"], record["summary"])

# iterate the cartesian product of model_grid x algorithm_grid x replicates
# task: "optimization" (uses n_restarts) or "sampling" (uses n_chains)
def run_grid(
    task,
    space,
    model_class,
    model_grid,
    algorithm_class,
    algorithm_grid,
    run_kwargs,
    alpha=2.0,
    lam=0.0,
    reg="linear",
    n_restarts=1,
    n_chains=1,
    keep_trace=False,
    keep_artifacts=False,
    experiment_name=None,
):
    records = []
    table = []
    # condition_id groups all replicates that share the same (model, algorithm) cell
    condition_id = 0

    for model_kwargs in parameter_grid(model_grid):
        for algorithm_kwargs in parameter_grid(algorithm_grid):
            # _disorder_id rides along on model_kwargs; pull it out for meta
            disorder_id = model_kwargs.get("_disorder_id")
            if task == "optimization":
                # one record per restart — independent fresh init each time
                for restart_id in range(int(n_restarts)):
                    record = run_single(
                        task=task,
                        space=space,
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        algorithm_class=algorithm_class,
                        algorithm_kwargs=algorithm_kwargs,
                        run_kwargs=run_kwargs,
                        alpha=alpha,
                        lam=lam,
                        reg=reg,
                        keep_trace=keep_trace,
                        keep_artifacts=keep_artifacts,
                        experiment_name=experiment_name,
                        chain_id=None,
                        restart_id=restart_id,
                        disorder_id=disorder_id,
                    )
                    record["meta"]["condition_id"] = condition_id
                    records.append(record)
                    table.append(flatten_record(record))
            elif task == "sampling":
                # one record per chain — used as replicas for q_ab and rhat
                for chain_id in range(int(n_chains)):
                    record = run_single(
                        task=task,
                        space=space,
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        algorithm_class=algorithm_class,
                        algorithm_kwargs=algorithm_kwargs,
                        run_kwargs=run_kwargs,
                        alpha=alpha,
                        lam=lam,
                        reg=reg,
                        keep_trace=keep_trace,
                        keep_artifacts=keep_artifacts,
                        experiment_name=experiment_name,
                        chain_id=chain_id,
                        restart_id=None,
                        disorder_id=disorder_id,
                    )
                    record["meta"]["condition_id"] = condition_id
                    records.append(record)
                    table.append(flatten_record(record))
            else:
                raise ValueError("task must be 'optimization' or 'sampling'")
            condition_id += 1

    return {"records": records, "table": table}

# prefix every key with <prefix>_ to namespace meta columns
def _flatten_prefixed(prefix, d):
    out = {}
    for key, value in d.items():
        out[f"{prefix}_{key}"] = value
    return out

# strip non-serializable junk (hamiltonian, callables) from run_kwargs for meta
def _clean_run_kwargs(run_kwargs):
    out = {}
    for key, value in run_kwargs.items():
        # discrete_hamiltonian is a live object — drop it from the meta row
        if key == "discrete_hamiltonian":
            continue
        # callables (schedules, hooks) get reduced to their __name__
        if callable(value):
            out[key] = getattr(value, "__name__", "callable")
        else:
            out[key] = value
    return out
