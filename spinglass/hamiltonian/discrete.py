# discrete spin hamiltonian H(s) = -1/2 s^T J s on s in {-1,+1}^n;
# exposes energy, local fields, and single/all-site flip deltas
import numpy as np

from ..utils.spin import ColumnCache

# thin wrapper around a model's coupling matrix J; cached column lookups
# accelerate sparse single-site updates in metropolis/gibbs
class DiscreteHamiltonian:
    def __init__(self, model):
        self.model = model
        self.J = model.J
        self._column_cache = None

    # lazily build the column cache; only sweepers that need per-column
    # access pay the construction cost
    def column_cache(self):
        if self._column_cache is None:
            self._column_cache = ColumnCache(self.J)
        return self._column_cache

    # total energy E(s) = -1/2 s^T J s; J symmetric with zero diagonal
    def energy(self, s):
        s = np.asarray(s, dtype=np.float64)
        Js = self.J @ s
        return -0.5 * float(s @ Js)

    # h_i = sum_j J_ij s_j; ravel keeps output 1d for sparse matrices
    def local_fields(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.asarray(self.J @ s).ravel()

    # delta E for flipping site i: dE = 2 s_i h_i (J symmetric, zero diag)
    def delta_energy(self, s, i, h=None):
        if h is None:
            h = self.local_fields(s)
        return 2.0 * float(s[i]) * float(h[i])

    # vectorized dE for all single-site flips; reuse precomputed h to skip Js
    def delta_energy_all(self, s, h=None):
        if h is None:
            h = self.local_fields(s)
        return 2.0 * np.asarray(s, dtype=np.float64) * h

    # m = (1/n) sum_i s_i
    @staticmethod
    def magnetization(s):
        return float(np.mean(s))
