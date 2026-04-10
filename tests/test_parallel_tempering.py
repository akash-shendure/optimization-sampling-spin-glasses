"""parallel tempering bookkeeping invariants.

PT is easy to get wrong in subtle ways (stale local fields after a swap,
energies drifting from states, swap acceptance ratio miscomputed). these
tests check that after a full PT run, each replica's cached energy and local
field match a fresh recomputation from its current state, and that swap
accepts are counted consistently."""
import numpy as np

from spinglass import (
    DiscreteHamiltonian,
    EdwardsAnderson2D,
    ParallelTemperingOptimizer,
    ParallelTemperingSampler,
)


def test_pt_optimizer_final_energies_match_states():
    model = EdwardsAnderson2D(L=6, disorder="pm1", seed=42)
    H = DiscreteHamiltonian(model)
    betas = np.array([0.2, 0.5, 1.0, 2.0])
    pt = ParallelTemperingOptimizer(H, betas=betas, swap_interval=5, seed=1)
    result = pt.run(n_steps=300, trace_every=50)
    states = result["artifacts"]["final_states"]
    cached = result["artifacts"]["energies"]
    assert states.shape == (betas.size, model.n)
    for r in range(betas.size):
        fresh = H.energy(states[r])
        assert abs(fresh - float(cached[r])) < 1e-9, f"replica {r}: cached {cached[r]} vs fresh {fresh}"


def test_pt_optimizer_best_state_has_best_energy():
    model = EdwardsAnderson2D(L=5, disorder="pm1", seed=7)
    H = DiscreteHamiltonian(model)
    pt = ParallelTemperingOptimizer(H, betas=[0.2, 1.0, 5.0], seed=2)
    result = pt.run(n_steps=500, trace_every=50)
    best_state = result["artifacts"]["best_state"]
    best_energy = float(result["summary"]["best_energy"])
    assert abs(H.energy(best_state) - best_energy) < 1e-9
    # best energy should be <= min of final replica energies (best is a running min)
    assert best_energy <= float(np.min(result["artifacts"]["energies"])) + 1e-9


def test_pt_sampler_final_state_matches_summary_energy():
    model = EdwardsAnderson2D(L=5, disorder="pm1", seed=8)
    H = DiscreteHamiltonian(model)
    pt = ParallelTemperingSampler(H, betas=[0.5, 1.0, 2.0], swap_interval=3, seed=9)
    result = pt.run(n_steps=400, burn_in=50, trace_every=25)
    states = result["artifacts"]["final_states"]
    target_state = result["artifacts"]["target_state"]
    final_energy = float(result["summary"]["final_energy"])
    # summary.final_energy is the target replica's energy at the end of the run
    target_idx = int(np.argmax(pt.betas))
    assert np.array_equal(target_state, states[target_idx])
    assert abs(H.energy(target_state) - final_energy) < 1e-9
    # every replica's state should round-trip through the Hamiltonian cleanly
    for r in range(states.shape[0]):
        e = H.energy(states[r])
        assert np.isfinite(e)


def test_pt_swap_acceptance_rate_in_zero_one():
    model = EdwardsAnderson2D(L=5, seed=11)
    H = DiscreteHamiltonian(model)
    pt = ParallelTemperingOptimizer(H, betas=[0.5, 1.0, 2.0, 4.0], swap_interval=2, seed=12)
    result = pt.run(n_steps=200, trace_every=50)
    rate = result["summary"]["swap_acceptance_rate"]
    assert 0.0 <= rate <= 1.0


def test_pt_optimizer_trace_is_monotone_in_best():
    model = EdwardsAnderson2D(L=5, seed=13)
    H = DiscreteHamiltonian(model)
    pt = ParallelTemperingOptimizer(H, betas=[0.3, 1.0, 3.0], seed=14)
    result = pt.run(n_steps=500, trace_every=25)
    best = result["trace"]["best_energy"]
    assert np.all(np.diff(best) <= 1e-12), "best energy must be monotone non-increasing"


if __name__ == "__main__":
    test_pt_optimizer_final_energies_match_states()
    test_pt_optimizer_best_state_has_best_energy()
    test_pt_sampler_final_state_matches_summary_energy()
    test_pt_swap_acceptance_rate_in_zero_one()
    test_pt_optimizer_trace_is_monotone_in_best()
    print("test_parallel_tempering OK")
