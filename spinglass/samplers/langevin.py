"""unadjusted Langevin updates on the relaxed space."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng


class LangevinSampler:
    def __init__(self, hamiltonian, beta, step_size, seed=None):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.step_size = float(step_size)
        self.seed = seed
        self.rng = make_rng(seed)

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
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)
        energy, grad = self.hamiltonian.energy_and_grad(x)

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
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(x.copy())
            if step == n_steps:
                break
            noise = self.rng.normal(size=self.model.n)
            x = x - self.step_size * self.beta * grad + np.sqrt(2.0 * self.step_size) * noise
            energy, grad = self.hamiltonian.energy_and_grad(x)

        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "langevin",
            "task": "sampling",
            "space": "relaxed",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "n_kept_samples": len(kept),
            "seed": self.seed,
        }
        artifacts = {"final_state": x}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.float64)
        if project:
            artifacts["projected_state"] = self.hamiltonian.project(x)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}

