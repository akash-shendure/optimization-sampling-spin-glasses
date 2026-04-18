"""Sherrington-Kirkpatrick dense mean-field couplings."""
import numpy as np

from ..utils.rng import make_rng


# fully connected, J_ij ~ N(0, 1/n) so that H is extensive
def build_sk_couplings(n, scale=None):
    rng = make_rng()
    # default variance 1/n gives the standard SK normalization
    sigma = (1.0 / np.sqrt(n)) if scale is None else scale
    A = rng.normal(0.0, sigma, size=(n, n))
    J = np.triu(A, k=1)
    J = J + J.T
    # triu(k=1) already zeros the diagonal, but be explicit
    np.fill_diagonal(J, 0.0)
    return J
