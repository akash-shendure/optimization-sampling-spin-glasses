# 2d periodic square-lattice adjacency and coupling builders (ferromagnet + edwards-anderson)
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng

# row,col -> flat index with wrap-around; modulo handles periodic boundary
def _flat(r, c, L):
    return (r % L) * L + (c % L)

# enumerate nearest-neighbor edges on an LxL grid
# returns (i, j) arrays with i < j so each edge appears once
def build_lattice_edges(L, periodic=True):
    edges_i = []
    edges_j = []
    # walk every site and emit its right and down neighbors
    for r in range(L):
        for c in range(L):
            u = _flat(r, c, L)
            # right neighbor; skipped at the boundary when not periodic
            if periodic or c + 1 < L:
                v = _flat(r, c + 1, L)
                a, b = (u, v) if u < v else (v, u)  # canonical order
                edges_i.append(a); edges_j.append(b)
            # down neighbor
            if periodic or r + 1 < L:
                v = _flat(r + 1, c, L)
                a, b = (u, v) if u < v else (v, u)
                edges_i.append(a); edges_j.append(b)
    return np.array(edges_i, dtype=np.int64), np.array(edges_j, dtype=np.int64)

# pack edge list + weights into a symmetric sparse J with zero diagonal
def _edges_to_sym_csr(n, ei, ej, w):
    # duplicate each (i,j,w) as (j,i,w) so J = J^T
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([w, w])
    J = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    J.setdiag(0.0)  # no self-interaction
    J.eliminate_zeros()
    return J

# 2d ising ferromagnet: all bonds equal to J0 > 0
def ferromagnet_couplings(L, J0=1.0, periodic=True):
    n = L * L
    ei, ej = build_lattice_edges(L, periodic=periodic)
    w = np.full(ei.shape, J0, dtype=np.float64)
    return _edges_to_sym_csr(n, ei, ej, w)

# edwards-anderson: same lattice topology but bonds are random (+/-1 or gaussian)
def ea_couplings(L, disorder="pm1", scale=1.0, periodic=True):
    rng = make_rng()
    n = L * L
    ei, ej = build_lattice_edges(L, periodic=periodic)
    m = ei.shape[0]
    # disorder on each bond — sign frustration is what makes the model glassy
    if disorder == "pm1":
        w = rng.choice(np.array([-1.0, 1.0]), size=m) * scale
    elif disorder == "gaussian":
        w = rng.normal(0.0, scale, size=m)
    else:
        raise ValueError(f"unknown disorder: {disorder}")
    return _edges_to_sym_csr(n, ei, ej, w)
