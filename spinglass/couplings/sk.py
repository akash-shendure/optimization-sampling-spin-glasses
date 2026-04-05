"""Sherrington-Kirkpatrick dense mean-field couplings."""
import numpy as np

from ..utils.rng import make_rng


# SK: fully connected, J_ij ~ N(0, 1/n) so that H is extensive
# return a dense symmetric numpy array with zero diagonal
def build_sk_couplings(n, scale=None, seed=None):
    rng = make_rng(seed)
    # default variance 1/n gives the standard SK normalization
    sigma = (1.0 / np.sqrt(n)) if scale is None else scale
    A = rng.normal(0.0, sigma, size=(n, n))
    # symmetrize: upper triangle copied across the diagonal
    J = np.triu(A, k=1)
    J = J + J.T
    # diagonal already zero from triu(k=1); be explicit anyway
    np.fill_diagonal(J, 0.0)
    return J
