# continuous relaxation H_tilde(x) = -sum J_ij t_i t_j + lam * R(t) with
# t = tanh(alpha x); smooth surrogate that ula/mala/hmc can drive
import numpy as np

# wraps a discrete model and exposes a smooth energy/grad on x in R^n
class RelaxedHamiltonian:
    def __init__(self, model, alpha=1.0, lam=0.0):
        self.model = model
        self.J = model.J
        self.alpha = float(alpha)
        self.lam = float(lam)

    # t = tanh(alpha x); 1 - t^2 = sech^2(alpha x) is reused below
    def _t(self, x):
        return np.tanh(self.alpha * np.asarray(x, dtype=np.float64))

    # smooth surrogate energy: pairwise interaction + non-binary penalty
    def energy(self, x):
        t = self._t(x)
        Jt = self.J @ t
        interact = -0.5 * float(t @ Jt)  # J symmetric with zero diag
        penalty = self.lam * float(np.sum(1.0 - t * t))
        return interact + penalty

    # grad of -0.5 t^T J t     = -alpha (1-t^2) (J t)
    # grad of lam sum(1-t^2)   = -2 alpha lam (1-t^2) t
    # grad of lam sum(1-t^2)^2 = -4 alpha lam (1-t^2)^2 t
    def grad(self, x):
        t = self._t(x)
        Jt = np.asarray(self.J @ t).ravel()
        sech2 = 1.0 - t * t
        return -self.alpha * sech2 * (Jt + 2.0 * self.lam * t)

    # combined call — samplers usually want both, this avoids two tanh evals
    def energy_and_grad(self, x):
        t = self._t(x)
        Jt = np.asarray(self.J @ t).ravel()
        sech2 = 1.0 - t * t
        interact = -0.5 * float(t @ Jt)
        penalty = self.lam * float(np.sum(sech2))
        g = -self.alpha * sech2 * (Jt + 2.0 * self.lam * t)
        return interact + penalty, g

    # round x back to s in {-1, +1}; map exact zeros to +1 as a tiebreak
    @staticmethod
    def project(x):
        s = np.sign(np.asarray(x, dtype=np.float64)).astype(np.int8)
        s[s == 0] = 1
        return s
