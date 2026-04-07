"""aggregate benchmark summaries for experiment outputs."""
from collections import defaultdict

import numpy as np

from ..diagnostics.mcmc_stats import ess, integrated_autocorr_time, rhat


def summarize_optimization_table(table, group_by, target_col=None):
    groups = _group_rows(table, group_by)
    out = []
    for key, rows in groups.items():
        best = np.asarray([row["best_energy"] for row in rows], dtype=np.float64)
        final = np.asarray([row["final_energy"] for row in rows], dtype=np.float64)
        runtime = np.asarray([row["runtime_sec"] for row in rows], dtype=np.float64)
        summary = _key_dict(group_by, key)
        summary.update(
            {
                "n_runs": len(rows),
                "mean_best_energy": float(np.mean(best)),
                "std_best_energy": float(np.std(best)),
                "min_best_energy": float(np.min(best)),
                "mean_final_energy": float(np.mean(final)),
                "std_final_energy": float(np.std(final)),
                "mean_runtime_sec": float(np.mean(runtime)),
                "std_runtime_sec": float(np.std(runtime)),
            }
        )

        if "best_projected_energy" in rows[0]:
            vals = np.asarray(
                [row["best_projected_energy"] for row in rows if row.get("best_projected_energy") is not None],
                dtype=np.float64,
            )
            if vals.size:
                summary["mean_best_projected_energy"] = float(np.mean(vals))
                summary["std_best_projected_energy"] = float(np.std(vals))

        if "hit_target" in rows[0]:
            hit = np.asarray([bool(row.get("hit_target", False)) for row in rows], dtype=np.float64)
            summary["success_rate"] = float(np.mean(hit))
            hit_steps = [row.get("hit_step") for row in rows if row.get("hit_step") is not None]
            hit_times = [row.get("hit_time_sec") for row in rows if row.get("hit_time_sec") is not None]
            if hit_steps:
                summary["mean_hit_step"] = float(np.mean(hit_steps))
            if hit_times:
                summary["mean_hit_time_sec"] = float(np.mean(hit_times))

        if target_col is not None and target_col in rows[0]:
            target_vals = np.asarray([row[target_col] for row in rows], dtype=np.float64)
            summary[f"mean_{target_col}"] = float(np.mean(target_vals))

        out.append(summary)
    return out


def collect_chain_traces(records, observable="energy", group_by=None):
    if group_by is None:
        group_by = ["condition_id"]
    grouped = defaultdict(list)
    for record in records:
        trace = record.get("trace")
        if trace is None or observable not in trace:
            continue
        key = tuple(record["meta"].get(name) for name in group_by)
        grouped[key].append((record["meta"], np.asarray(trace[observable], dtype=np.float64)))

    out = []
    for key, items in grouped.items():
        lengths = [arr.size for _, arr in items]
        if not lengths:
            continue
        t = min(lengths)
        chains = np.stack([arr[:t] for _, arr in items], axis=0)
        payload = _key_dict(group_by, key)
        payload["observable"] = observable
        payload["n_chains"] = chains.shape[0]
        payload["n_draws"] = chains.shape[1]
        payload["chains"] = chains
        payload["metas"] = [meta for meta, _ in items]
        out.append(payload)
    return out


def summarize_sampling_table(table, records=None, group_by=None, observable="energy"):
    if group_by is None:
        group_by = ["condition_id"]
    groups = _group_rows(table, group_by)
    out = []

    trace_map = {}
    if records is not None:
        for payload in collect_chain_traces(records, observable=observable, group_by=group_by):
            key = tuple(payload[name] for name in group_by)
            trace_map[key] = payload

    for key, rows in groups.items():
        runtime = np.asarray([row["runtime_sec"] for row in rows], dtype=np.float64)
        final = np.asarray([row["final_energy"] for row in rows], dtype=np.float64)
        mean_energy = np.asarray([row.get("mean_energy", np.nan) for row in rows], dtype=np.float64)
        summary = _key_dict(group_by, key)
        summary.update(
            {
                "n_runs": len(rows),
                "mean_final_energy": float(np.mean(final)),
                "std_final_energy": float(np.std(final)),
                "mean_runtime_sec": float(np.mean(runtime)),
                "std_runtime_sec": float(np.std(runtime)),
                "mean_energy": float(np.nanmean(mean_energy)),
            }
        )

        accept = np.asarray([row.get("acceptance_rate", np.nan) for row in rows], dtype=np.float64)
        if not np.all(np.isnan(accept)):
            summary["mean_acceptance_rate"] = float(np.nanmean(accept))

        swap = np.asarray([row.get("swap_acceptance_rate", np.nan) for row in rows], dtype=np.float64)
        if not np.all(np.isnan(swap)):
            summary["mean_swap_acceptance_rate"] = float(np.nanmean(swap))

        payload = trace_map.get(key)
        if payload is not None and payload["n_chains"] >= 2:
            chains = payload["chains"]
            ess_val = ess(chains)
            rhat_val = rhat(chains)
            tau = integrated_autocorr_time(chains)
            summary[f"{observable}_ess"] = float(ess_val)
            summary[f"{observable}_rhat"] = float(rhat_val)
            summary[f"{observable}_tau_int"] = float(tau)
            total_runtime = float(np.sum(runtime))
            if total_runtime > 0:
                summary[f"{observable}_ess_per_sec"] = float(ess_val / total_runtime)
            summary["n_chains"] = int(payload["n_chains"])
            summary["n_draws"] = int(payload["n_draws"])

        out.append(summary)
    return out


def _group_rows(rows, group_by):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in group_by)
        grouped[key].append(row)
    return grouped


def _key_dict(names, key):
    return {name: value for name, value in zip(names, key)}

