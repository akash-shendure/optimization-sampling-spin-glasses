"""observable helpers for spin configurations."""
import numpy as np


def magnetization(states):
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        return float(np.mean(s))
    return np.mean(s, axis=-1)


def overlap(state_a, state_b):
    a = np.asarray(state_a, dtype=np.float64)
    b = np.asarray(state_b, dtype=np.float64)
    return np.mean(a * b, axis=-1)


def pairwise_overlaps(states):
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("states must have shape (n_chains, n_spins)")
    n = s.shape[0]
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            q = float(np.mean(s[i] * s[j]))
            out[i, j] = q
            out[j, i] = q
    return out

