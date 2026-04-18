# sparse erdos-renyi spin-glass couplings: random graph with mean connectivity c, random bond weights
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng

# bernoulli-sample edges from the strict upper triangle at probability p
def _sample_er_edges(n, p, rng):
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.shape[0]) < p
    return iu[mask], ju[mask]

# build a symmetric sparse J on an erdos-renyi graph
# c is target mean degree; p = c/(n-1) gives expected degree c
def build_erdos_renyi_couplings(n, c=3.0, disorder="gaussian", scale=1.0):
    rng = make_rng()
    p = min(1.0, c / max(n - 1, 1))  # clamp for tiny n where c/(n-1) > 1
    ei, ej = _sample_er_edges(n, p, rng)
    m = ei.shape[0]
    # assign a random bond weight to each sampled edge
    if disorder == "pm1":
        w = rng.choice(np.array([-1.0, 1.0]), size=m) * scale
    elif disorder == "gaussian":
        w = rng.normal(0.0, scale, size=m)
    else:
        raise ValueError(f"unknown disorder: {disorder}")
    # mirror (i,j) -> (j,i) so the matrix is symmetric
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([w, w])
    J = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    J.setdiag(0.0)  # no self-interaction
    J.eliminate_zeros()
    return J
