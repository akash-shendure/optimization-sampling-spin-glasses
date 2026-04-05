# dense sherrington-kirkpatrick couplings: fully-connected J with N(0, 1/n) or +/-1/sqrt(n) bonds
import numpy as np

from ..utils.rng import make_rng

def build_sk_couplings(n, scale=None, seed=None):
    rng = make_rng(seed)
    sigma = (1.0 / np.sqrt(n)) if scale is None else scale
    A = rng.normal(0.0, sigma, size=(n, n))
    J = np.triu(A, k=1)
    J = J + J.T
    np.fill_diagonal(J, 0.0)  # no self-interaction
    return J
