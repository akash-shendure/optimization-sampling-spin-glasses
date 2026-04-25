# dense sherrington-kirkpatrick couplings: fully-connected J with N(0, 1/n) or +/-1/sqrt(n) bonds
import numpy as np

from ..utils.rng import make_rng

# build a dense symmetric J for the SK model on n sites
# default sigma = 1/sqrt(n) gives the standard SK normalization (O(1) energy per site)
def build_sk_couplings(n, scale=None, disorder="gaussian"):
    rng = make_rng()
    sigma = (1.0 / np.sqrt(n)) if scale is None else scale
    # sample only the strict upper triangle, then mirror — guarantees symmetry
    iu, ju = np.triu_indices(n, k=1)
    m = iu.shape[0]
    if disorder == "gaussian":
        w = rng.normal(0.0, sigma, size=m)
    elif disorder == "pm1":
        w = rng.choice(np.array([-1.0, 1.0]), size=m) * sigma
    else:
        raise ValueError(f"unknown disorder: {disorder}")
    # fill upper triangle and symmetrize
    J = np.zeros((n, n), dtype=np.float64)
    J[iu, ju] = w
    J = J + J.T
    np.fill_diagonal(J, 0.0)  # no self-interaction
    return J
