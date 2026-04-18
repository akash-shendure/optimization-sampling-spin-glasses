"""single-run and sweep execution helpers."""
from copy import deepcopy

from .builders import build_algorithm, build_hamiltonian, build_model
from .grids import merge_dicts, parameter_grid
from ..hamiltonian.discrete import DiscreteHamiltonian


def run_single(
    task,
    space,
    model_class,
    model_kwargs,
    algorithm_class,
    algorithm_kwargs,
    run_kwargs,
    alpha=1.0,
    lam=0.0,
    keep_trace=True,
    keep_artifacts=False,
    experiment_name=None,
    chain_id=None,
    restart_id=None,
    disorder_id=None,
):
    model_kwargs = deepcopy(model_kwargs)
    algorithm_kwargs = deepcopy(algorithm_kwargs)
    run_kwargs = deepcopy(run_kwargs)

    # _disorder_id is a grid-fanout marker, not a model constructor kwarg
    model_kwargs.pop("_disorder_id", None)

    model = build_model(model_class, **model_kwargs)
    hamiltonian = build_hamiltonian(model, space=space, alpha=alpha, lam=lam)
    algorithm = build_algorithm(algorithm_class, hamiltonian, **algorithm_kwargs)

    if space == "relaxed" and "project" not in run_kwargs:
        run_kwargs["project"] = True
    if space == "relaxed" and run_kwargs.get("project", False):
        run_kwargs.setdefault("discrete_hamiltonian", DiscreteHamiltonian(model))

    result = algorithm.run(**run_kwargs)

    meta = {
        "experiment_name": experiment_name,
        "task": task,
        "space": space,
        "model_name": model.__class__.__name__,
        "algorithm_name": algorithm_class.__name__,
        "chain_id": chain_id,
        "restart_id": restart_id,
        "disorder_id": disorder_id,
        "alpha": float(alpha) if space == "relaxed" else None,
        "lam": float(lam) if space == "relaxed" else None,
    }
    meta.update(_flatten_prefixed("model", model.describe()))
    meta.update(_flatten_prefixed("algorithm", algorithm_kwargs))
    meta.update(_flatten_prefixed("run", _clean_run_kwargs(run_kwargs)))

    return {
        "meta": meta,
        "summary": result["summary"],
        "trace": result["trace"] if keep_trace else None,
        "artifacts": result["artifacts"] if keep_artifacts else None,
    }


def flatten_record(record):
    return merge_dicts(record["meta"], record["summary"])


def run_grid(
    task,
    space,
    model_class,
    model_grid,
    algorithm_class,
    algorithm_grid,
    run_kwargs,
    alpha=1.0,
    lam=0.0,
    n_restarts=1,
    n_chains=1,
    keep_trace=False,
    keep_artifacts=False,
    experiment_name=None,
):
    records = []
    table = []
    condition_id = 0

    for model_kwargs in parameter_grid(model_grid):
        for algorithm_kwargs in parameter_grid(algorithm_grid):
            disorder_id = model_kwargs.get("_disorder_id")
            if task == "optimization":
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


def _flatten_prefixed(prefix, d):
    out = {}
    for key, value in d.items():
        out[f"{prefix}_{key}"] = value
    return out


def _clean_run_kwargs(run_kwargs):
    out = {}
    for key, value in run_kwargs.items():
        if key == "discrete_hamiltonian":
            continue
        if callable(value):
            out[key] = getattr(value, "__name__", "callable")
        else:
            out[key] = value
    return out

