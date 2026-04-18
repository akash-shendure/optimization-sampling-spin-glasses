# parallel tempering: n_replica metropolis chains along a beta ladder with replica swaps
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields_fast

# adjacent-pair swap accept prob = min(1, exp((beta_r - beta_{r+1})(E_{r+1} - E_r)))
# helps cold replica escape barriers via excursions to hot replicas
class ParallelTemperingSampler:
    # betas: 1d ascending or descending ladder; target_index picks which replica we report
    def __init__(self, hamiltonian, betas, swap_interval=1, target_index=None):
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1 or betas.size < 2:
            raise ValueError("betas must be a 1d array with at least two entries")
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.betas = betas
        self.swap_interval = int(swap_interval)
        # default target is the coldest replica (largest beta) — usually the ground-state hunter
        self.target_index = int(np.argmax(betas) if target_index is None else target_index)
        self.rng = make_rng()

    # run all replicas in lockstep; one MH step per replica per outer step, swaps every swap_interval
    def run(self, states0=None, n_steps=1000, burn_in=0, thin=1, trace_every=1, store_samples=False):
        # init per-replica state, cached fields, energies
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

        # main loop
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

            # per-replica metropolis step at its own beta
            for r, beta in enumerate(self.betas):
                i = int(self.rng.integers(self.model.n))
                dE = self.hamiltonian.delta_energy(states[r], i, h=fields[r])
                if dE <= 0.0 or self.rng.random() < np.exp(-beta * dE):
                    states[r, i] = -states[r, i]
                    update_local_fields_fast(fields[r], cache, i, states[r, i])
                    energies[r] += float(dE)
                    accept_count += 1

            # propose adjacent-pair swaps; d >= 0 always accept (detailed-balance shortcut)
            if self.swap_interval > 0 and (step + 1) % self.swap_interval == 0:
                for r in range(n_replica - 1):
                    swap_attempts += 1
                    dbeta = self.betas[r] - self.betas[r + 1]
                    d = dbeta * (energies[r + 1] - energies[r])
                    if d >= 0.0 or self.rng.random() < np.exp(d):
                        # swap states, cached fields, and energies in lockstep
                        states[[r, r + 1]] = states[[r + 1, r]]
                        fields[[r, r + 1]] = fields[[r + 1, r]]
                        energies[[r, r + 1]] = energies[[r + 1, r]]
                        swap_accepts += 1

        # finalize trace + summary, including swap diagnostics
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
        }
        artifacts = {"final_states": states, "target_state": states[self.target_index].copy()}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.int8)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}
