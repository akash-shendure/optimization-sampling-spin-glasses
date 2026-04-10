"""Hamiltonian Monte Carlo on the relaxed space.

leapfrog integration of (x, p) under potential U(x) = beta * H_tilde(x) and
kinetic K(p) = 0.5 * p^T p (unit mass). accept/reject on the joint energy
H = U + K preserves exp(-U(x)) marginally, so x targets exp(-beta H_tilde)."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng


class HMCSampler:
    def __init__(self, hamiltonian, beta, step_size, n_leapfrog=10, seed=None):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.step_size = float(step_size)
        self.n_leapfrog = int(n_leapfrog)
        self.seed = seed
        self.rng = make_rng(seed)

    # U(x) = beta * H_tilde(x); grad_U(x) = beta * grad_H_tilde(x)
    def _potential_and_grad(self, x):
        energy, grad = self.hamiltonian.energy_and_grad(x)
        return self.beta * energy, self.beta * grad

    # one leapfrog trajectory of length n_leapfrog
    def _leapfrog(self, x, p, grad_U):
        eps = self.step_size
        p = p - 0.5 * eps * grad_U
        for i in range(self.n_leapfrog):
            x = x + eps * p
            _, grad_U = self._potential_and_grad(x)
            if i < self.n_leapfrog - 1:
                p = p - eps * grad_U
        p = p - 0.5 * eps * grad_U
        # negate for reversibility — irrelevant for symmetric K but keeps math honest
        return x, -p

    def run(
        self,
        x0=None,
        n_steps=1000,
        burn_in=0,
        thin=1,
        trace_every=1,
        store_samples=False,
        project=False,
        discrete_hamiltonian=None,
    ):
        x = self.rng.normal(size=self.model.n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        U, grad_U = self._potential_and_grad(x)
        energy = U / self.beta  # raw H_tilde for reporting
        accept_count = 0
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)

        for step in range(n_steps + 1):
            elapsed = now() - start
            projected_energy = np.nan
            if project and discrete_hamiltonian is not None:
                projected_energy = discrete_hamiltonian.energy(self.hamiltonian.project(x))
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    grad_norm=float(np.linalg.norm(grad_U) / max(self.beta, 1e-12)),
                    projected_energy=projected_energy,
                    acceptance_rate=accept_count / max(1, step),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(x.copy())
            if step == n_steps:
                break
            p0 = self.rng.normal(size=self.model.n)
            H_old = U + 0.5 * float(p0 @ p0)
            x_new, p_new = self._leapfrog(x, p0, grad_U)
            U_new, grad_U_new = self._potential_and_grad(x_new)
            H_new = U_new + 0.5 * float(p_new @ p_new)
            log_alpha = H_old - H_new  # accept if new joint energy lower
            if log_alpha >= 0.0 or np.log(self.rng.random()) < log_alpha:
                x = x_new
                U = U_new
                grad_U = grad_U_new
                energy = U / self.beta
                accept_count += 1

        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "hmc",
            "task": "sampling",
            "space": "relaxed",
            "n_steps": n_steps,
            "n_leapfrog": self.n_leapfrog,
            "step_size": self.step_size,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "acceptance_rate": accept_count / max(1, n_steps),
            "n_kept_samples": len(kept),
            "seed": self.seed,
        }
        artifacts = {"final_state": x}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.float64)
        if project:
            artifacts["projected_state"] = self.hamiltonian.project(x)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}
