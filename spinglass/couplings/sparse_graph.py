"""sparse random-graph Ising glass couplings (Erdos-Renyi G(n, p))."""
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng


def _sample_er_edges(n, p, rng):
    # direct upper-triangular sample; swap to geometric-gap if n grows past O(n^2) memory
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.shape[0]) < p
    return iu[mask], ju[mask]


# c is target mean degree, so p = c/(n-1)
def build_erdos_renyi_couplings(n, c=3.0, disorder="gaussian", scale=1.0):
    rng = make_rng()
    p = min(1.0, c / max(n - 1, 1))
    ei, ej = _sample_er_edges(n, p, rng)
    m = ei.shape[0]
    if disorder == "pm1":
        w = rng.choice(np.array([-1.0, 1.0]), size=m) * scale
    elif disorder == "gaussian":
        w = rng.normal(0.0, scale, size=m)
    else:
        raise ValueError(f"unknown disorder: {disorder}")
    # symmetrize into CSR with zero diagonal
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([w, w])
    J = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    J.setdiag(0.0)
    J.eliminate_zeros()
    return J
