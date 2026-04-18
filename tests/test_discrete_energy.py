"""discrete Hamiltonian energy invariants.

checks:
  - DiscreteHamiltonian.energy vs explicit double-sum over edges
  - delta_energy vs fresh-recompute after an actual flip
  - delta_energy_all vs per-site delta_energy
  - local_fields vs explicit J @ s
tested across the lattice, sparse ER, and dense SK model families so both
sparse-CSR and dense-ndarray J paths are exercised."""
import numpy as np

from spinglass import (
    DiscreteHamiltonian,
    EdwardsAnderson2D,
    IsingFerromagnet2D,
    SherringtonKirkpatrick,
    SparseRandomGlass,
)


def _brute_force_energy(J, s):
    # evaluates H = -sum_{i<j} J_ij s_i s_j directly from J
    J_dense = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    n = s.size
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total -= J_dense[i, j] * float(s[i]) * float(s[j])
    return total


def _assert_close(a, b, atol=1e-10, what=""):
    assert abs(float(a) - float(b)) <= atol, f"{what}: {a} vs {b}"


def _sample_models():
    return [
        IsingFerromagnet2D(L=5),
        EdwardsAnderson2D(L=5, disorder="pm1"),
        EdwardsAnderson2D(L=4, disorder="gaussian"),
        SparseRandomGlass(n=24, c=3.0),
        SherringtonKirkpatrick(n=20),
    ]


def test_energy_matches_brute_force():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        for _ in range(3):
            s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
            _assert_close(H.energy(s), _brute_force_energy(H.J, s), atol=1e-9, what=f"energy {type(model).__name__}")


def test_local_fields_match_matvec():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
        h = H.local_fields(s)
        J_dense = H.J.toarray() if hasattr(H.J, "toarray") else np.asarray(H.J)
        expected = J_dense @ s.astype(np.float64)
        assert np.allclose(h, expected, atol=1e-10)


def test_delta_energy_matches_fresh_recompute():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
        h = H.local_fields(s)
        for _ in range(5):
            i = int(rng.integers(model.n))
            dE = H.delta_energy(s, i, h=h)
            E_before = H.energy(s)
            s_flipped = s.copy()
            s_flipped[i] = -s_flipped[i]
            E_after = H.energy(s_flipped)
            _assert_close(dE, E_after - E_before, atol=1e-9, what=f"delta_energy site {i}")


def test_delta_energy_all_matches_loop():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
        h = H.local_fields(s)
        dE_vec = H.delta_energy_all(s, h=h)
        for i in range(model.n):
            _assert_close(dE_vec[i], H.delta_energy(s, i, h=h), atol=1e-12)


def test_magnetization_and_zero_diag():
    # sanity on the validated invariants J has zero diag and is symmetric
    for model in _sample_models():
        J_dense = model.J.toarray() if hasattr(model.J, "toarray") else np.asarray(model.J)
        assert np.allclose(np.diag(J_dense), 0.0)
        assert np.allclose(J_dense, J_dense.T, atol=1e-12)
    s = np.array([1, 1, -1, -1], dtype=np.int8)
    assert DiscreteHamiltonian.magnetization(s) == 0.0


if __name__ == "__main__":
    test_energy_matches_brute_force()
    test_local_fields_match_matvec()
    test_delta_energy_matches_fresh_recompute()
    test_delta_energy_all_matches_loop()
    test_magnetization_and_zero_diag()
    print("test_discrete_energy OK")
