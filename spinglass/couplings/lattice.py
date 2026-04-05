"""2D square-lattice edge list + coupling matrices (ferromagnet and EA)."""
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng


# map 2D grid index (r,c) to flat index on an LxL torus
def _flat(r, c, L):
    return (r % L) * L + (c % L)


# enumerate unique undirected edges on an LxL torus with periodic BCs
# returns (i, j) arrays with i < j
def build_lattice_edges(L, periodic=True):
    edges_i = []
    edges_j = []
    for r in range(L):
        for c in range(L):
            u = _flat(r, c, L)
            # right neighbor
            if periodic or c + 1 < L:
                v = _flat(r, c + 1, L)
                a, b = (u, v) if u < v else (v, u)
                edges_i.append(a); edges_j.append(b)
            # down neighbor
            if periodic or r + 1 < L:
                v = _flat(r + 1, c, L)
                a, b = (u, v) if u < v else (v, u)
                edges_i.append(a); edges_j.append(b)
    return np.array(edges_i, dtype=np.int64), np.array(edges_j, dtype=np.int64)


# build a symmetric sparse coupling matrix from edges + per-edge weights
# zero diagonal; CSR for fast matvec (used everywhere downstream)
def _edges_to_sym_csr(n, ei, ej, w):
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([w, w])
    J = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    J.setdiag(0.0)  # no self-interaction
    J.eliminate_zeros()
    return J


# uniform J_ij = +J0 ferromagnet on the lattice
def ferromagnet_couplings(L, J0=1.0, periodic=True):
    n = L * L
    ei, ej = build_lattice_edges(L, periodic=periodic)
    w = np.full(ei.shape, J0, dtype=np.float64)
    return _edges_to_sym_csr(n, ei, ej, w)


# Edwards-Anderson: same lattice edges but each J_ij iid in {-1,+1} (or Gaussian)
def ea_couplings(L, disorder="pm1", scale=1.0, seed=None, periodic=True):
    rng = make_rng(seed)
    n = L * L
    ei, ej = build_lattice_edges(L, periodic=periodic)
    m = ei.shape[0]
    if disorder == "pm1":
        w = rng.choice(np.array([-1.0, 1.0]), size=m) * scale
    elif disorder == "gaussian":
        w = rng.normal(0.0, scale, size=m)
    else:
        raise ValueError(f"unknown disorder: {disorder}")
    return _edges_to_sym_csr(n, ei, ej, w)
