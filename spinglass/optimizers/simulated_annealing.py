"""simulated annealing via single-spin Metropolis updates."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast


class SimulatedAnnealing:
    def __init__(self, hamiltonian, beta_schedule):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta_schedule = beta_schedule
        self.rng = make_rng()

    def _beta(self, step, n_steps):
        if callable(self.beta_schedule):
            return float(self.beta_schedule(step, n_steps))
        schedule = np.asarray(self.beta_schedule, dtype=np.float64)
        if schedule.ndim != 1 or schedule.size == 0:
            raise ValueError("beta_schedule must be callable or a non-empty 1d array")
        idx = min(step, schedule.size - 1)
        return float(schedule[idx])

    def run(self, s0=None, n_steps=1000, trace_every=1, target_energy=None, store_states=False):
        s = self.model.random_state(self.rng) if s0 is None else np.asarray(s0, dtype=np.int8).copy()
        h = self.hamiltonian.local_fields(s)
        energy = self.hamiltonian.energy(s)
        cache = self.hamiltonian.column_cache()
        best_energy = energy
        best_state = s.copy()
        accept_count = 0
        hit_step = None
        hit_time = None
        trace = init_trace()
        state_trace = []
        start = now()
        n_steps = int(n_steps)

        for step in range(n_steps + 1):
            elapsed = now() - start
            beta = self._beta(step, n_steps)
            if step == 0 or step % trace_every == 0:
                accept_rate = accept_count / max(1, step)
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    best_energy=best_energy,
                    beta=beta,
                    acceptance_rate=accept_rate,
                    magnetization=self.hamiltonian.magnetization(s),
                )
                if store_states:
                    state_trace.append(s.copy())
            if target_energy is not None and hit_step is None and best_energy <= target_energy:
                hit_step = step
                hit_time = elapsed
            if step == n_steps:
                break
            i = int(self.rng.integers(self.model.n))
            dE = self.hamiltonian.delta_energy(s, i, h=h)
            if dE <= 0.0 or self.rng.random() < np.exp(-beta * dE):
                s[i] = -s[i]
                update_local_fields_fast(h, cache, i, s[i])
                energy += float(dE)
                accept_count += 1
                if energy < best_energy:
                    best_energy = energy
                    best_state = s.copy()

        summary = {
            "algorithm": "simulated_annealing",
            "task": "optimization",
            "space": "discrete",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "best_energy": float(best_energy),
            "acceptance_rate": accept_count / max(1, n_steps),
            "hit_target": hit_step is not None,
            "hit_step": hit_step,
            "hit_time_sec": hit_time,
        }
        artifacts = {"final_state": s, "best_state": best_state}
        if store_states:
            artifacts["state_trace"] = np.asarray(state_trace, dtype=np.int8)
        return {"summary": summary, "trace": finalize_trace(trace), "artifacts": artifacts}

