# unadjusted langevin (ULA) on relaxed x in R^n; x += -beta*grad*step + sqrt(2 step)*z
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng

# no MH correction — bias-for-speed; samples target proportional to exp(-beta H_tilde) only as step -> 0
class LangevinSampler:
    # hamiltonian must expose energy_and_grad(x); step_size is the euler-maruyama dt
    def __init__(self, hamiltonian, beta, step_size):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.step_size = float(step_size)
        self.rng = make_rng()

    # project=True logs discrete energy of sign(x) using the supplied discrete hamiltonian
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
        # init x and one energy_and_grad call we reuse across iter boundaries
        x = self.rng.normal(size=self.model.n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)
        energy, grad = self.hamiltonian.energy_and_grad(x)

        # main ULA loop; one extra iter at the end so trace records final state
        for step in range(n_steps + 1):
            elapsed = now() - start
            projected_energy = np.nan
            if project and discrete_hamiltonian is not None:
                projected_energy = discrete_hamiltonian.energy(self.hamiltonian.project(x))
            if step == 0 or step % trace_every == 0:
                s_proj = self.hamiltonian.project(x)
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    grad_norm=float(np.linalg.norm(grad)),
                    projected_energy=projected_energy,
                    magnetization=float(np.mean(s_proj)),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(x.copy())
            if step == n_steps:
                break
            # euler-maruyama: drift -beta*grad*dt + diffusion sqrt(2 dt) N(0,I)
            noise = self.rng.normal(size=self.model.n)
            x = x - self.step_size * self.beta * grad + np.sqrt(2.0 * self.step_size) * noise
            energy, grad = self.hamiltonian.energy_and_grad(x)

        # finalize trace + summary; optionally emit projected discrete state
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
        }
        artifacts = {"final_state": x}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.float64)
        if project:
            artifacts["projected_state"] = self.hamiltonian.project(x)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}
