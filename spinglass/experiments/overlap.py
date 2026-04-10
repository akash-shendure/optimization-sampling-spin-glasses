# replica-overlap diagnostics: q_ab between independent runs, histograms, mixing
from collections import defaultdict

import numpy as np

from ..diagnostics.mcmc_stats import ess, integrated_autocorr_time, rhat
from ..diagnostics.observables import overlap

# normalize an array to a {-1,+1} vector — relaxed states get sign()ed
def _as_spin_vector(arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return a
    # already ±1 within float tolerance — pass through unchanged
    if np.all(np.abs(np.abs(a) - 1.0) < 1e-12):
        return a
    s = np.sign(a)
    s[s == 0] = 1.0  # tiebreak zero -> +1, same convention as projector
    return s

# gather per-group spin states from records, looking in artifacts for state_key
def collect_replica_states(records, group_by=None, state_key=None):
    if group_by is None:
        group_by = ["condition_id"]
    # prefer projected_state when state_key not specified — relaxed runs need projection
    candidate_keys = [state_key] if state_key is not None else ["projected_state", "final_state"]
    grouped = defaultdict(list)
    for record in records:
        artifacts = record.get("artifacts") or {}
        state = None
        # take first available candidate key — order matters
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
        # need at least two replicas to form any pair overlap
        if len(items) < 2:
            continue
        states = np.stack([arr for _, arr in items], axis=0)
        payload = {name: value for name, value in zip(group_by, key)}
        payload["n_replicas"] = states.shape[0]
        payload["n_spins"] = states.shape[1]
        payload["states"] = states
        payload["metas"] = [meta for meta, _ in items]
        out.append(payload)
    return out

# all pairwise overlaps q_ab = (1/n) sum_i s^a_i s^b_i for a<b
def replica_overlap_values(states):
    s = np.asarray(states, dtype=np.float64)
    n_rep = s.shape[0]
    if n_rep < 2:
        return np.zeros(0, dtype=np.float64)
    qs = []
    # upper-triangular loop avoids double counting and q_aa=1 self-pairs
    for a in range(n_rep):
        for b in range(a + 1, n_rep):
            qs.append(float(overlap(s[a], s[b])))
    return np.asarray(qs, dtype=np.float64)

# per-group mean/std/min/max of pairwise overlaps — replica-symmetry probe
def summarize_replica_overlaps(records, group_by=None, state_key="final_state"):
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
                # |q| matters for glassy phases where q can be bimodal around 0
                "mean_abs_q": float(np.mean(np.abs(qs))),
                "std_q": float(np.std(qs)),
                "min_q": float(np.min(qs)),
                "max_q": float(np.max(qs)),
            }
        )
        out.append(summary)
    return out

# discretized P(q) — non-trivial structure flags replica-symmetry breaking
def overlap_histogram(records, bins=21, group_by=None, state_key="final_state"):
    # fixed [-1,1] support so histograms are comparable across groups
    edges = np.linspace(-1.0, 1.0, int(bins) + 1)
    payloads = collect_replica_states(records, group_by=group_by, state_key=state_key)
    out = []
    for payload in payloads:
        qs = replica_overlap_values(payload["states"])
        counts, _ = np.histogram(qs, bins=edges)
        summary = {name: payload[name] for name in (group_by or ["condition_id"])}
        summary["bin_edges"] = edges
        summary["counts"] = counts
        # guard divide-by-zero when no pairs — keep density well-defined
        summary["density"] = counts / max(1, counts.sum())
        summary["n_pairs"] = int(qs.size)
        out.append(summary)
    return out

# trace-level overlap q_{c,c+1}(t) between adjacent chains — diagnoses sampler mixing
def collect_overlap_chain_traces(records, chain_pair_key="chain_id", group_by=None):
    if group_by is None:
        group_by = ["condition_id"]
    grouped = defaultdict(list)
    for record in records:
        artifacts = record.get("artifacts") or {}
        samples = artifacts.get("samples")
        if samples is None:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        # expect (n_draws, n_spins); skip anything else
        if arr.ndim != 2:
            continue
        # project to ±1 if relaxed samples leaked through
        if not np.all(np.abs(np.abs(arr) - 1.0) < 1e-12):
            arr = np.sign(arr)
            arr[arr == 0] = 1.0
        key = tuple(record["meta"].get(name) for name in group_by)
        grouped[key].append((record["meta"].get(chain_pair_key), arr))

    out = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        # sort by chain_id so pair p is always (chain p, chain p+1)
        items.sort(key=lambda pair: (pair[0] if pair[0] is not None else 0))
        lengths = [arr.shape[0] for _, arr in items]
        # truncate to shortest trace so all rows align in time
        t = min(lengths)
        n_pairs = len(items) - 1
        chain_q = np.zeros((n_pairs, t), dtype=np.float64)
        # per-step overlap between adjacent chains — sliding pairs, not full pairwise
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

# ess/rhat/tau on the overlap trace itself — does q mix across replicas?
def summarize_overlap_mixing(records, group_by=None):
    payloads = collect_overlap_chain_traces(records, group_by=group_by)
    out = []
    for payload in payloads:
        chains = payload["chains"]
        # need >=2 pairs and a few draws for diagnostics to be meaningful
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
