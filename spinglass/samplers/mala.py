"""Metropolis-adjusted Langevin on the relaxed space."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng


class MALASampler:
    def __init__(self, hamiltonian, beta, step_size):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.step_size = float(step_size)
        self.rng = make_rng()

    def _log_target(self, x):
        return -self.beta * self.hamiltonian.energy(x)

    def _proposal_mean(self, x, grad):
        return x - self.step_size * self.beta * grad

    def _log_q(self, to_x, from_x, from_grad):
        mean = self._proposal_mean(from_x, from_grad)
        resid = to_x - mean
        return -0.25 * float(resid @ resid) / self.step_size

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
        energy, grad = self.hamiltonian.energy_and_grad(x)
        logp = -self.beta * energy
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
                    grad_norm=float(np.linalg.norm(grad)),
                    projected_energy=projected_energy,
                    acceptance_rate=accept_count / max(1, step),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(x.copy())
            if step == n_steps:
                break
            mean = self._proposal_mean(x, grad)
            proposal = mean + np.sqrt(2.0 * self.step_size) * self.rng.normal(size=self.model.n)
            prop_energy, prop_grad = self.hamiltonian.energy_and_grad(proposal)
            prop_logp = -self.beta * prop_energy
            log_alpha = (
                prop_logp
                + self._log_q(x, proposal, prop_grad)
                - logp
                - self._log_q(proposal, x, grad)
            )
            if log_alpha >= 0.0 or np.log(self.rng.random()) < log_alpha:
                x = proposal
                energy = prop_energy
                grad = prop_grad
                logp = prop_logp
                accept_count += 1

        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "mala",
            "task": "sampling",
            "space": "relaxed",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "acceptance_rate": accept_count / max(1, n_steps),
            "n_kept_samples": len(kept),
        }
        artifacts = {"final_state": x}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.float64)
        if project:
            artifacts["projected_state"] = self.hamiltonian.project(x)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}
