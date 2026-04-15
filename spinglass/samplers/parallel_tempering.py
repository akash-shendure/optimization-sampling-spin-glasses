"""parallel tempering sampler for the discrete Hamiltonian."""
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast


class ParallelTemperingSampler:
    def __init__(self, hamiltonian, betas, swap_interval=1, target_index=None, seed=None):
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1 or betas.size < 2:
            raise ValueError("betas must be a 1d array with at least two entries")
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.betas = betas
        self.swap_interval = int(swap_interval)
        self.target_index = int(np.argmax(betas) if target_index is None else target_index)
        self.seed = seed
        self.rng = make_rng(seed)

    def run(self, states0=None, n_steps=1000, burn_in=0, thin=1, trace_every=1, store_samples=False):
        n_replica = self.betas.size
        if states0 is None:
            states = np.asarray([self.model.random_state(self.rng) for _ in range(n_replica)], dtype=np.int8)
        else:
            states = np.asarray(states0, dtype=np.int8).copy()
        fields = np.asarray([self.hamiltonian.local_fields(s) for s in states], dtype=np.float64)
        energies = np.asarray([self.hamiltonian.energy(s) for s in states], dtype=np.float64)
        cache = self.hamiltonian.column_cache()
        accept_count = 0
        swap_attempts = 0
        swap_accepts = 0
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)

        for step in range(n_steps + 1):
            elapsed = now() - start
            target_energy = float(energies[self.target_index])
            target_state = states[self.target_index]
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=target_energy,
                    mean_energy=float(np.mean(energies)),
                    min_energy=float(np.min(energies)),
                    magnetization=self.hamiltonian.magnetization(target_state),
                    acceptance_rate=accept_count / max(1, step * n_replica),
                    swap_acceptance_rate=swap_accepts / max(1, swap_attempts),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(target_state.copy())
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

        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "parallel_tempering_sampler",
            "task": "sampling",
            "space": "discrete",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energies[self.target_index]),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "acceptance_rate": accept_count / max(1, n_steps * n_replica),
            "swap_acceptance_rate": swap_accepts / max(1, swap_attempts),
            "n_kept_samples": len(kept),
            "seed": self.seed,
        }
        artifacts = {"final_states": states, "target_state": states[self.target_index].copy()}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.int8)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}

