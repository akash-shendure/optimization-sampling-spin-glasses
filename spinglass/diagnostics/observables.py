# physical observables: magnetization m and replica overlaps q_{ab}
import numpy as np

# m = (1/n) sum s_i; supports a single state or a batch with spins on the last axis
def magnetization(states):
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        return float(np.mean(s))
    return np.mean(s, axis=-1)

# overlap q_{ab} = (1/n) sum s_i^a s_i^b between two replicas (or batched pairs)
def overlap(state_a, state_b):
    a = np.asarray(state_a, dtype=np.float64)
    b = np.asarray(state_b, dtype=np.float64)
    return np.mean(a * b, axis=-1)

# full symmetric overlap matrix Q[i,j] = q(s_i, s_j) for an array of replicas
def pairwise_overlaps(states):
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("states must have shape (n_chains, n_spins)")
    n = s.shape[0]
    # diagonal is q(s, s) = 1 since s_i^2 = 1 for binary spins
    out = np.eye(n, dtype=np.float64)
    # fill upper triangle then mirror; q is symmetric
    for i in range(n):
        for j in range(i + 1, n):
            q = float(np.mean(s[i] * s[j]))
            out[i, j] = q
            out[j, i] = q
    return out
