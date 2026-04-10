"""overlap-focused benchmark summaries for glassy models.

glassy models don't have a symmetry-breaking order parameter like magnetization,
so the overlap q = (1/n) sum s^(a)_i s^(b)_i between replicas is the natural
observable for diagnosing sampling in the glassy phase. these helpers work on
the output of experiments.runner.run_grid when records carry store_samples=True
or a per-chain spin snapshot is attached."""
from collections import defaultdict

import numpy as np

from ..diagnostics.mcmc_stats import ess, integrated_autocorr_time, rhat
from ..diagnostics.observables import overlap


def _as_spin_vector(arr):
    """coerce a recorded state to a ±1 spin vector.

    discrete samplers already store int8 spins; relaxed samplers store the raw x
    in R^n and additionally (when project=True) a projected_state. for uniform
    downstream handling we project anything non-binary through sign(.)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return a
    # treat values that are already in {-1, +1} as-is; otherwise project
    if np.all(np.abs(np.abs(a) - 1.0) < 1e-12):
        return a
    s = np.sign(a)
    s[s == 0] = 1.0
    return s


def collect_replica_states(records, group_by=None, state_key=None):
    """group recorded per-chain final states across chains within each condition.

    tries projected_state first, then final_state, unless state_key is given.
    float states are projected to ±1 via sign(.) so overlaps stay in [-1, 1]."""
    if group_by is None:
        group_by = ["condition_id"]
    candidate_keys = [state_key] if state_key is not None else ["projected_state", "final_state"]
    grouped = defaultdict(list)
    for record in records:
        artifacts = record.get("artifacts") or {}
        state = None
        for name in candidate_keys:
            if name in artifacts and artifacts[name] is not None:
                state = artifacts[name]
                break
        if state is None:
            continue
        arr = _as_spin_vector(state)
        if arr.ndim != 1 or arr.size == 0:
            continue
        key = tuple(record["meta"].get(name) for name in group_by)
        grouped[key].append((record["meta"], arr))

    out = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue  # need at least two replicas for an overlap
        states = np.stack([arr for _, arr in items], axis=0)
        payload = {name: value for name, value in zip(group_by, key)}
        payload["n_replicas"] = states.shape[0]
        payload["n_spins"] = states.shape[1]
        payload["states"] = states
        payload["metas"] = [meta for meta, _ in items]
        out.append(payload)
    return out


def replica_overlap_values(states):
    """all pairwise overlaps q_ab among replicas (upper triangle), as a 1d array."""
    s = np.asarray(states, dtype=np.float64)
    n_rep = s.shape[0]
    if n_rep < 2:
        return np.zeros(0, dtype=np.float64)
    qs = []
    for a in range(n_rep):
        for b in range(a + 1, n_rep):
            qs.append(float(overlap(s[a], s[b])))
    return np.asarray(qs, dtype=np.float64)


def summarize_replica_overlaps(records, group_by=None, state_key="final_state"):
    """per-condition summary of the final-state overlap distribution.

    reports mean |q|, mean q, std q, min/max q, and the number of replica pairs
    contributing. this is a coarse but useful phase diagnostic — at high T the
    overlap distribution concentrates near 0, in an ordered phase near +-1, and
    in a glassy phase is broadly supported."""
    payloads = collect_replica_states(records, group_by=group_by, state_key=state_key)
    out = []
    for payload in payloads:
        qs = replica_overlap_values(payload["states"])
        summary = {name: payload[name] for name in (group_by or ["condition_id"])}
        summary.update(
            {
                "n_replicas": int(payload["n_replicas"]),
                "n_pairs": int(qs.size),
                "mean_q": float(np.mean(qs)),
                "mean_abs_q": float(np.mean(np.abs(qs))),
                "std_q": float(np.std(qs)),
                "min_q": float(np.min(qs)),
                "max_q": float(np.max(qs)),
            }
        )
        out.append(summary)
    return out


def overlap_histogram(records, bins=21, group_by=None, state_key="final_state"):
    """per-condition histogram of final-state overlaps on [-1, 1]."""
    edges = np.linspace(-1.0, 1.0, int(bins) + 1)
    payloads = collect_replica_states(records, group_by=group_by, state_key=state_key)
    out = []
    for payload in payloads:
        qs = replica_overlap_values(payload["states"])
        counts, _ = np.histogram(qs, bins=edges)
        summary = {name: payload[name] for name in (group_by or ["condition_id"])}
        summary["bin_edges"] = edges
        summary["counts"] = counts
        summary["density"] = counts / max(1, counts.sum())
        summary["n_pairs"] = int(qs.size)
        out.append(summary)
    return out


def collect_overlap_chain_traces(records, chain_pair_key="chain_id", group_by=None):
    """build (n_chain_pairs, n_draws) trace arrays of q between adjacent chains
    within each condition — lets us feed overlap into ESS / R-hat directly.

    requires records whose artifacts hold a "samples" field of shape
    (n_draws, n_spins) (set store_samples=True in the sampler run). relaxed
    samples are projected to ±1 before computing overlaps."""
    if group_by is None:
        group_by = ["condition_id"]
    grouped = defaultdict(list)
    for record in records:
        artifacts = record.get("artifacts") or {}
        samples = artifacts.get("samples")
        if samples is None:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2:
            continue
        if not np.all(np.abs(np.abs(arr) - 1.0) < 1e-12):
            arr = np.sign(arr)
            arr[arr == 0] = 1.0
        key = tuple(record["meta"].get(name) for name in group_by)
        grouped[key].append((record["meta"].get(chain_pair_key), arr))

    out = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda pair: (pair[0] if pair[0] is not None else 0))
        lengths = [arr.shape[0] for _, arr in items]
        t = min(lengths)
        n_pairs = len(items) - 1
        chain_q = np.zeros((n_pairs, t), dtype=np.float64)
        for p in range(n_pairs):
            a = items[p][1][:t]
            b = items[p + 1][1][:t]
            chain_q[p] = np.mean(a * b, axis=1)
        payload = {name: value for name, value in zip(group_by, key)}
        payload["chains"] = chain_q
        payload["n_chain_pairs"] = int(n_pairs)
        payload["n_draws"] = int(t)
        out.append(payload)
    return out


def summarize_overlap_mixing(records, group_by=None):
    """ESS / R-hat / tau_int on overlap trajectories between replica chains.

    high-level phase diagnostic for the glassy regime: if overlap between
    independent chains mixes well you have convergence; if it is stuck, the
    sampler is trapped in a single basin."""
    payloads = collect_overlap_chain_traces(records, group_by=group_by)
    out = []
    for payload in payloads:
        chains = payload["chains"]
        if chains.shape[0] < 2 or chains.shape[1] < 4:
            continue
        summary = {name: payload[name] for name in (group_by or ["condition_id"])}
        summary["n_chain_pairs"] = int(chains.shape[0])
        summary["n_draws"] = int(chains.shape[1])
        summary["q_ess"] = float(ess(chains))
        summary["q_rhat"] = float(rhat(chains))
        summary["q_tau_int"] = float(integrated_autocorr_time(chains))
        summary["mean_q"] = float(np.mean(chains))
        summary["std_q"] = float(np.std(chains))
        out.append(summary)
    return out
