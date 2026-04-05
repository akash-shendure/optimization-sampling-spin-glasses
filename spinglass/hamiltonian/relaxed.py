"""continuous relaxation H_tilde(x) = -sum J_ij t_i t_j + lambda sum (1 - t_i^2),
with t_i = tanh(alpha x_i). used by gradient descent / Adam / Langevin / MALA / HMC."""
import numpy as np


class RelaxedHamiltonian:
    # alpha controls sharpness of the tanh squashing; lambda penalizes non-binary t
    def __init__(self, model, alpha=1.0, lam=0.0):
        self.model = model
        self.J = model.J
        self.alpha = float(alpha)
        self.lam = float(lam)

    # squashed variables t = tanh(alpha x) — cached together with x for gradient reuse
    def _t(self, x):
        return np.tanh(self.alpha * np.asarray(x, dtype=np.float64))

    # smooth surrogate energy
    def energy(self, x):
        t = self._t(x)
        Jt = self.J @ t
        interact = -0.5 * float(t @ Jt)  # symmetric, zero diag
        penalty = self.lam * float(np.sum(1.0 - t * t))
        return interact + penalty

    # grad_k = -a(1-t_k^2) * ((Jt)_k + 2 lam t_k)
    def grad(self, x):
        t = self._t(x)
        Jt = np.asarray(self.J @ t).ravel()
        sech2 = 1.0 - t * t  # = sech^2(alpha x)
        return -self.alpha * sech2 * (Jt + 2.0 * self.lam * t)

    # convenience: one call returns (energy, grad) — samplers often want both
    def energy_and_grad(self, x):
        t = self._t(x)
        Jt = np.asarray(self.J @ t).ravel()
        sech2 = 1.0 - t * t
        interact = -0.5 * float(t @ Jt)
        penalty = self.lam * float(np.sum(sech2))
        g = -self.alpha * sech2 * (Jt + 2.0 * self.lam * t)
        return interact + penalty, g

    # map exact zeros to +1 as a consistent tiebreak
    @staticmethod
    def project(x):
        s = np.sign(np.asarray(x, dtype=np.float64)).astype(np.int8)
        s[s == 0] = 1
        return s
