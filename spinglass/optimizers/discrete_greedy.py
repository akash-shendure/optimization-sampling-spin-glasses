"""greedy single-spin descent for the discrete Hamiltonian."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast


class GreedySpinDescent:
    def __init__(self, hamiltonian, proposal="best", seed=None):
        if proposal not in {"best", "random"}:
            raise ValueError("proposal must be 'best' or 'random'")
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.proposal = proposal
        self.seed = seed
        self.rng = make_rng(seed)

    def run(self, s0=None, n_steps=None, trace_every=1, target_energy=None, store_states=False):
        s = self.model.random_state(self.rng) if s0 is None else np.asarray(s0, dtype=np.int8).copy()
        h = self.hamiltonian.local_fields(s)
        energy = self.hamiltonian.energy(s)
        cache = self.hamiltonian.column_cache()
        best_energy = energy
        best_state = s.copy()
        hit_step = None
        hit_time = None
        start = now()
        trace = init_trace()
        state_trace = []
        n_max = self.model.n if n_steps is None else int(n_steps)

        for step in range(n_max + 1):
            elapsed = now() - start
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    best_energy=best_energy,
                    magnetization=self.hamiltonian.magnetization(s),
                )
                if store_states:
                    state_trace.append(s.copy())
            if target_energy is not None and hit_step is None and best_energy <= target_energy:
                hit_step = step
                hit_time = elapsed
            if step == n_max:
                break
            dE = self.hamiltonian.delta_energy_all(s, h=h)
            negative = np.flatnonzero(dE < 0.0)
            if negative.size == 0:
                break
            if self.proposal == "best":
                i = int(np.argmin(dE))
            else:
                i = int(self.rng.choice(negative))
            energy += float(dE[i])
            s[i] = -s[i]
            update_local_fields_fast(h, cache, i, s[i])
            if energy < best_energy:
                best_energy = energy
                best_state = s.copy()

        summary = {
            "algorithm": "greedy_spin_descent",
            "task": "optimization",
            "space": "discrete",
            "n_steps": int(trace["step"][-1]) if trace else 0,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "best_energy": float(best_energy),
            "hit_target": hit_step is not None,
            "hit_step": hit_step,
            "hit_time_sec": hit_time,
            "seed": self.seed,
        }
        artifacts = {"final_state": s, "best_state": best_state}
        if store_states:
            artifacts["state_trace"] = np.asarray(state_trace, dtype=np.int8)
        return {"summary": summary, "trace": finalize_trace(trace), "artifacts": artifacts}

