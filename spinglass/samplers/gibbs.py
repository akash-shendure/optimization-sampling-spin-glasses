# heat-bath / gibbs sampler on s in {-1,+1}^n; p(s_i=+1 | rest) = sigmoid(2 beta h_i)
import numpy as np

from ..utils.records import append_trace, finalize_trace, init_trace, now
from ..utils.rng import make_rng
from ..utils.spin import update_local_fields

# always resamples s_i from its full conditional — no reject step
class GibbsSampler:
    def __init__(self, hamiltonian, beta, seed=None):
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.beta = float(beta)
        self.seed = seed
        self.rng = make_rng(seed)

    # n_steps single-site gibbs sweeps with optional sample retention
    def run(self, s0=None, n_steps=1000, burn_in=0, thin=1, trace_every=1, store_samples=False):
        # init state, cached fields, running energy
        s = self.model.random_state(self.rng) if s0 is None else np.asarray(s0, dtype=np.int8).copy()
        h = self.hamiltonian.local_fields(s)
        energy = self.hamiltonian.energy(s)
        kept = []
        trace = init_trace()
        start = now()
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        thin = int(thin)

        # main loop; extra iter at the end to log the final state
        for step in range(n_steps + 1):
            elapsed = now() - start
            if step == 0 or step % trace_every == 0:
                append_trace(
                    trace,
                    step=step,
                    time_sec=elapsed,
                    energy=energy,
                    magnetization=self.hamiltonian.magnetization(s),
                )
            if step >= burn_in and (step - burn_in) % thin == 0 and store_samples:
                kept.append(s.copy())
            if step == n_steps:
                break
            # heat-bath update: p(+1) = 1/(1+exp(-2 beta h_i)) is the sigmoid form
            i = int(self.rng.integers(self.model.n))
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * self.beta * h[i]))
            s_new = np.int8(1 if self.rng.random() < p_plus else -1)
            # only do incremental field update when site actually changed sign
            if s_new != s[i]:
                dE = self.hamiltonian.delta_energy(s, i, h=h)
                s[i] = s_new
                update_local_fields(h, self.hamiltonian.J, i, s[i])
                energy += float(dE)

        # finalize: pack trace and summary stats
        trace_out = finalize_trace(trace)
        summary = {
            "algorithm": "gibbs",
            "task": "sampling",
            "space": "discrete",
            "n_steps": n_steps,
            "runtime_sec": now() - start,
            "final_energy": float(energy),
            "mean_energy": float(np.mean(trace_out["energy"])),
            "n_kept_samples": len(kept),
            "seed": self.seed,
        }
        artifacts = {"final_state": s}
        if store_samples:
            artifacts["samples"] = np.asarray(kept, dtype=np.int8)
        return {"summary": summary, "trace": trace_out, "artifacts": artifacts}
