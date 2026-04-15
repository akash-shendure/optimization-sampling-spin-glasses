"""discrete Hamiltonian H(s) = -sum_{i<j} J_ij s_i s_j on s in {-1,+1}^n."""
import numpy as np

from ..utils.spin import ColumnCache


# model = disorder realization, Hamiltonian = physics ops
class DiscreteHamiltonian:
    def __init__(self, model):
        self.model = model
        self.J = model.J  # alias — sparse or dense, both support @
        self._column_cache = None

    # lazy shared cache — O(n*nnz_col) precompute paid once per disorder realization
    def column_cache(self):
        if self._column_cache is None:
            self._column_cache = ColumnCache(self.J)
        return self._column_cache

    # H = -0.5 s^T J s since J is symmetric with zero diagonal
    def energy(self, s):
        s = np.asarray(s, dtype=np.float64)
        Js = self.J @ s
        return -0.5 * float(s @ Js)

    # local fields h_i = sum_j J_ij s_j — used by Metropolis / Gibbs / greedy
    def local_fields(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.asarray(self.J @ s).ravel()

    # dE = 2 s_i h_i (the -s_i h_i term flips to +s_i h_i)
    def delta_energy(self, s, i, h=None):
        if h is None:
            h = self.local_fields(s)
        return 2.0 * float(s[i]) * float(h[i])

    # vectorized version: energy change if each site were flipped independently
    def delta_energy_all(self, s, h=None):
        if h is None:
            h = self.local_fields(s)
        return 2.0 * np.asarray(s, dtype=np.float64) * h

    # magnetization — cheap to expose here since sampling code needs it
    @staticmethod
    def magnetization(s):
        return float(np.mean(s))
