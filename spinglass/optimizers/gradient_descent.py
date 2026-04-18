"""gradient descent on the relaxed Hamiltonian."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng


class GradientDescentOptimizer:
    def __init__(self, hamiltonian, lr=1e-2):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.lr = float(lr)
        self.rng = make_rng()

    def run(
        self,
        x0=None,
        n_steps=1000,
        trace_every=1,
        target_energy=None,
        project=False,
        discrete_hamiltonian=None,
        store_states=False,
    ):
        x = self.rng.normal(size=self.model.n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        energy, grad = self.hamiltonian.energy_and_grad(x)
        best_energy = float(energy)
        best_x = x.copy()
        hit_step = None
        hit_time = None
        trace = init_trace()
        state_trace = []
        start = now()
        n_steps = int(n_steps)

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
                    best_energy=best_energy,
                    grad_norm=float(np.linalg.norm(grad)),
                    projected_energy=projected_energy,
                )
                if store_states:
                    state_trace.append(x.copy())
            if target_energy is not None and hit_step is None and best_energy <= target_energy:
                hit_step = step
                hit_time = elapsed
            if step == n_steps:
                break
            x -= self.lr * grad
            energy, grad = self.hamiltonian.energy_and_grad(x)
            if energy < best_energy:
                best_energy = float(energy)
                best_x = x.copy()

        final_projected_state = None
        final_projected_energy = None
        best_projected_state = None
        best_projected_energy = None
        if project:
            final_projected_state = self.hamiltonian.project(x)
            best_projected_state = self.hamiltonian.project(best_x)
            if discrete_hamiltonian is not None:
                final_projected_energy = float(discrete_hamiltonian.energy(final_projected_state))
                best_projected_energy = float(discrete_hamiltonian.energy(best_projected_state))

        summary = {
            "algorithm": "gradient_descent",
            "task": "optimization",
            "space": "relaxed",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "best_energy": float(best_energy),
            "final_projected_energy": final_projected_energy,
            "best_projected_energy": best_projected_energy,
            "hit_target": hit_step is not None,
            "hit_step": hit_step,
            "hit_time_sec": hit_time,
        }
        artifacts = {"final_state": x, "best_state": best_x}
        if project:
            artifacts["final_projected_state"] = final_projected_state
            artifacts["best_projected_state"] = best_projected_state
        if store_states:
            artifacts["state_trace"] = np.asarray(state_trace, dtype=np.float64)
        return {"summary": summary, "trace": finalize_trace(trace), "artifacts": artifacts}

