"""single-spin Metropolis sampling on the discrete state space."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast


class MetropolisSampler:
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.rng = make_rng()

    def run(self, s0=None, n_steps=1000, burn_in=0, thin=1, trace_every=1, store_samples=False):
        s = self.model.random_state(self.rng) if s0 is None else np.asarray(s0, dtype=np.int8).copy()
        h = self.hamiltonian.local_fields(s)
        energy = self.hamiltonian.energy(s)
        cache = self.hamiltonian.column_cache()
        accept_count = 0
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)

        for step in range(n_steps + 1):
            elapsed = now() - start
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    magnetization=self.hamiltonian.magnetization(s),
                    acceptance_rate=accept_count / max(1, step),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(s.copy())
            if step == n_steps:
                break
            i = int(self.rng.integers(self.model.n))
            dE = self.hamiltonian.delta_energy(s, i, h=h)
            if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
                s[i] = -s[i]
                update_local_fields_fast(h, cache, i, s[i])
                energy += float(dE)
                accept_count += 1

        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "metropolis",
            "task": "sampling",
            "space": "discrete",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "acceptance_rate": accept_count / max(1, n_steps),
            "n_kept_samples": len(kept),
        }
        artifacts = {"final_state": s}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.int8)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}

