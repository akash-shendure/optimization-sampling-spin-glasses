"""parallel tempering used as a search heuristic."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast


class ParallelTemperingOptimizer:
    def __init__(self, hamiltonian, betas, swap_interval=1):
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1 or betas.size < 2:
            raise ValueError("betas must be a 1d array with at least two entries")
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.betas = betas
        self.swap_interval = int(swap_interval)
        self.rng = make_rng()

    def run(self, states0=None, n_steps=1000, trace_every=1, target_energy=None, store_states=False):
        n_replica = self.betas.size
        if states0 is None:
            states = np.asarray([self.model.random_state(self.rng) for _ in range(n_replica)], dtype=np.int8)
        else:
            states = np.asarray(states0, dtype=np.int8).copy()
        fields = np.asarray([self.hamiltonian.local_fields(s) for s in states], dtype=np.float64)
        energies = np.asarray([self.hamiltonian.energy(s) for s in states], dtype=np.float64)
        cache = self.hamiltonian.column_cache()
        best_idx = int(np.argmin(energies))
        best_energy = float(energies[best_idx])
        best_state = states[best_idx].copy()
        accept_count = 0
        swap_attempts = 0
        swap_accepts = 0
        hit_step = None
        hit_time = None
        trace = init_trace()
        state_trace = []
        start = now()
        n_steps = int(n_steps)

        for step in range(n_steps + 1):
            elapsed = now() - start
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    best_energy=best_energy,
                    min_energy=float(np.min(energies)),
                    mean_energy=float(np.mean(energies)),
                    acceptance_rate=accept_count / max(1, step * n_replica),
                    swap_acceptance_rate=swap_accepts / max(1, swap_attempts),
                )
                if store_states:
                    state_trace.append(states.copy())
            if target_energy is not None and hit_step is None and best_energy <= target_energy:
                hit_step = step
                hit_time = elapsed
            if step == n_steps:
                break

            for r, beta in enumerate(self.betas):
                i = int(self.rng.integers(self.model.n))
                dE = self.hamiltonian.delta_energy(states[r], i, h=fields[r])
                if dE <= 0.0 or self.rng.random() < np.exp(-beta * dE):
                    states[r, i] = -states[r, i]
                    update_local_fields_fast(fields[r], cache, i, states[r, i])
                    energies[r] += float(dE)
                    accept_count += 1
                    if energies[r] < best_energy:
                        best_energy = float(energies[r])
                        best_state = states[r].copy()

            if self.swap_interval > 0 and (step + 1) % self.swap_interval == 0:
                for r in range(n_replica - 1):
                    swap_attempts += 1
                    dbeta = self.betas[r] - self.betas[r + 1]
                    d = dbeta * (energies[r + 1] - energies[r])
                    if d >= 0.0 or self.rng.random() < np.exp(d):
                        states[[r, r + 1]] = states[[r + 1, r]]
                        fields[[r, r + 1]] = fields[[r + 1, r]]
                        energies[[r, r + 1]] = energies[[r + 1, r]]
                        swap_accepts += 1

        summary = {
            "algorithm": "parallel_tempering_optimizer",
            "task": "optimization",
            "space": "discrete",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(np.min(energies)),
            "best_energy": float(best_energy),
            "acceptance_rate": accept_count / max(1, n_steps * n_replica),
            "swap_acceptance_rate": swap_accepts / max(1, swap_attempts),
            "hit_target": hit_step is not None,
            "hit_step": hit_step,
            "hit_time_sec": hit_time,
        }
        artifacts = {"final_states": states, "best_state": best_state, "energies": energies.copy()}
        if store_states:
            artifacts["state_trace"] = np.asarray(state_trace, dtype=np.int8)
        return {"summary": summary, "trace": finalize_trace(trace), "artifacts": artifacts}

