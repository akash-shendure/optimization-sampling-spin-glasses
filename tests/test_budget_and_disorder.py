"""Budget resolution and n_disorders fan-out invariants.

these tests check the matched-compute convention and the disorder-averaging
plumbing that the canonical study depends on. if these drift, benchmark
plots silently compare apples to oranges — worth catching fast."""
import numpy as np

from spinglass import Budget, budget_to_n_steps
from spinglass.experiments.budget import hamiltonian_evals, steps, sweeps
from spinglass.experiments.studies import _disorder_seeds, _inject_seeds, sampling_beta_sweep
from spinglass import IsingFerromagnet2D


def test_budget_steps_passes_through():
    b = steps(1234)
    assert b.to_n_steps(n_spins=50, space="discrete") == 1234
    assert b.to_n_steps(n_spins=50, space="relaxed") == 1234


def test_budget_sweeps_scales_discrete_by_n():
    b = sweeps(3)
    assert b.to_n_steps(n_spins=64, space="discrete") == 192  # 3 * 64
    # for relaxed, one grad step = one sweep, so no scaling
    assert b.to_n_steps(n_spins=64, space="relaxed") == 3


def test_budget_hamiltonian_evals_matches_compute():
    b = hamiltonian_evals(10)
    # 10 full H evaluations on 64 spins ~ 640 single-flip updates for discrete
    assert b.to_n_steps(n_spins=64, space="discrete") == 640
    # 10 full gradient steps for relaxed
    assert b.to_n_steps(n_spins=64, space="relaxed") == 10


def test_budget_rejects_bad_inputs():
    try:
        Budget(kind="nope", value=10)
    except ValueError:
        pass
    else:
        assert False, "Budget should reject unknown kind"
    try:
        Budget(kind="steps", value=-1)
    except ValueError:
        pass
    else:
        assert False, "Budget should reject non-positive value"


def test_budget_to_n_steps_accepts_plain_int():
    # legacy callers can still pass an int
    assert budget_to_n_steps(500, n_spins=16, space="discrete") == 500
    assert budget_to_n_steps(500, n_spins=16, space="relaxed") == 500


def test_disorder_seeds_are_distinct_and_ordered():
    seeds = _disorder_seeds(base_seed=7, n_disorders=5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    assert seeds == sorted(seeds)
    assert seeds[0] == 7


def test_inject_seeds_generates_n_disorders():
    out = _inject_seeds({"L": [8]}, n_disorders=4, base_seed=3)
    assert "seed" in out
    assert isinstance(out["seed"], list)
    assert len(out["seed"]) == 4
    assert len(set(out["seed"])) == 4


def test_inject_seeds_respects_caller_supplied_seeds():
    # if caller already supplied a list of seeds, leave it alone
    out = _inject_seeds({"L": [8], "seed": [10, 20, 30]}, n_disorders=5, base_seed=0)
    assert out["seed"] == [10, 20, 30]


def test_sampling_sweep_fans_over_disorders_and_averages():
    betas = [0.3, 0.8]
    # two disorder seeds, three chains each — summary should average across them
    res = sampling_beta_sweep(
        IsingFerromagnet2D,
        model_kwargs={"L": [6]},
        betas=betas,
        n_chains=2,
        n_steps=200,
        burn_in=40,
        trace_every=5,
        n_disorders=2,
        base_seed=13,
    )
    # 2 betas x 2 disorder seeds x 2 chains = 8 rows
    assert len(res["records"]) == 8
    # but grouped summary collapses by beta
    assert len(res["grouped"]) == 2
    assert res["n_disorders"] == 2
    # physical sanity: mean energy decreases with beta
    rows = sorted(res["grouped"], key=lambda r: r["algorithm_beta"])
    assert rows[0]["mean_energy"] > rows[-1]["mean_energy"]


def test_sampling_sweep_with_budget_honors_sweeps_convention():
    res = sampling_beta_sweep(
        IsingFerromagnet2D,
        model_kwargs={"L": [6]},
        betas=[0.5],
        n_chains=1,
        budget=sweeps(3),  # 3 sweeps x 36 spins = 108 effective steps
        burn_in=0,
        trace_every=1,
        n_disorders=1,
    )
    assert res["effective_n_steps"] == 3 * 36


if __name__ == "__main__":
    test_budget_steps_passes_through()
    test_budget_sweeps_scales_discrete_by_n()
    test_budget_hamiltonian_evals_matches_compute()
    test_budget_rejects_bad_inputs()
    test_budget_to_n_steps_accepts_plain_int()
    test_disorder_seeds_are_distinct_and_ordered()
    test_inject_seeds_generates_n_disorders()
    test_inject_seeds_respects_caller_supplied_seeds()
    test_sampling_sweep_fans_over_disorders_and_averages()
    test_sampling_sweep_with_budget_honors_sweeps_convention()
    print("test_budget_and_disorder OK")
