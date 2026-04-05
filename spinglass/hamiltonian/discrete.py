"""discrete Hamiltonian H(s) = -sum_{i<j} J_ij s_i s_j on s in {-1,+1}^n."""
import numpy as np


# thin wrapper around a SpinModel: all energy math lives here, not on the model
# keeps model = disorder realization, Hamiltonian = physics operations
class DiscreteHamiltonian:
    def __init__(self, model):
        self.model = model
        self.J = model.J  # alias — sparse or dense, both support @

    # full energy of a configuration s
    # H = -sum_{i<j} J_ij s_i s_j = -0.5 s^T J s  (J symmetric, zero diag)
    def energy(self, s):
        s = np.asarray(s, dtype=np.float64)
        Js = self.J @ s
        return -0.5 * float(s @ Js)

    # local fields h_i = sum_j J_ij s_j — used by Metropolis / Gibbs / greedy
    def local_fields(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.asarray(self.J @ s).ravel()

    # energy change from flipping site i: dE = 2 s_i h_i
    # (flipping i changes -s_i h_i term into +s_i h_i; factor of 2)
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
