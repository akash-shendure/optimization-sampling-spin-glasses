"""summary helpers for batched optimizer and sampler runs."""
import numpy as np


def summarize_optimizer_runs(results):
    if not results:
        return {}
    best = np.array([r["summary"]["best_energy"] for r in results], dtype=np.float64)
    runtime = np.array([r["summary"]["runtime_sec"] for r in results], dtype=np.float64)
    hit = np.array([bool(r["summary"].get("hit_target", False)) for r in results])
    out = {
        "n_runs": len(results),
        "mean_best_energy": float(np.mean(best)),
        "std_best_energy": float(np.std(best)),
        "mean_runtime_sec": float(np.mean(runtime)),
        "std_runtime_sec": float(np.std(runtime)),
        "success_rate": float(np.mean(hit)),
    }
    hit_times = [
        r["summary"].get("hit_time_sec")
        for r in results
        if r["summary"].get("hit_time_sec") is not None
    ]
    if hit_times:
        out["mean_hit_time_sec"] = float(np.mean(hit_times))
    return out


def summarize_sampler_runs(results):
    if not results:
        return {}
    runtime = np.array([r["summary"]["runtime_sec"] for r in results], dtype=np.float64)
    mean_energy = np.array([r["summary"].get("mean_energy", np.nan) for r in results], dtype=np.float64)
    accept = np.array([r["summary"].get("acceptance_rate", np.nan) for r in results], dtype=np.float64)
    out = {
        "n_runs": len(results),
        "mean_runtime_sec": float(np.mean(runtime)),
        "std_runtime_sec": float(np.std(runtime)),
        "mean_energy": float(np.nanmean(mean_energy)),
    }
    if not np.all(np.isnan(accept)):
        out["mean_acceptance_rate"] = float(np.nanmean(accept))
    return out
