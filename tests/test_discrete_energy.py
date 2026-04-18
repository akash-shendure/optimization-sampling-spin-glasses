# H(s) = -0.5 s^T J s consistency, local fields = J s, delta_energy = 2 s_i h_i, magnetization
import numpy as np

from spinglass import (
    DiscreteHamiltonian,
    EdwardsAnderson2D,
    IsingFerromagnet2D,
    SherringtonKirkpatrick,
    SparseRandomGlass,
)

# brute force ground truth: sum over upper triangle, -J_ij s_i s_j
def _brute_force_energy(J, s):
    J_dense = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    n = s.size
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total -= J_dense[i, j] * float(s[i]) * float(s[j])
    return total

def _assert_close(a, b, atol=1e-10, what=""):
    assert abs(float(a) - float(b)) <= atol, f"{what}: {a} vs {b}"

# cover sparse + dense, ferromagnetic + glassy, pm1 + gaussian disorder
def _sample_models():
    return [
        IsingFerromagnet2D(L=5),
        EdwardsAnderson2D(L=5, disorder="pm1"),
        EdwardsAnderson2D(L=4, disorder="gaussian"),
        SparseRandomGlass(n=24, c=3.0),
        SherringtonKirkpatrick(n=20),
    ]

# H.energy must equal the O(n^2) double loop on random spin configs
def test_energy_matches_brute_force():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        for _ in range(3):
            s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
            # 1e-9 atol: brute sum accumulates n*(n-1)/2 float ops
            _assert_close(H.energy(s), _brute_force_energy(H.J, s), atol=1e-9, what=f"energy {type(model).__name__}")

# local_fields h_i = sum_j J_ij s_j must equal a plain J @ s matvec
def test_local_fields_match_matvec():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
        h = H.local_fields(s)
        J_dense = H.J.toarray() if hasattr(H.J, "toarray") else np.asarray(H.J)
        expected = J_dense @ s.astype(np.float64)
        assert np.allclose(h, expected, atol=1e-10)

# delta_energy(i) using cached h must equal E(flipped) - E(current)
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

# vectorized delta_energy_all must agree element-wise with the per-site call
def test_delta_energy_all_matches_loop():
    rng = np.random.default_rng()
    for model in _sample_models():
        H = DiscreteHamiltonian(model)
        s = 2 * rng.integers(0, 2, size=model.n).astype(np.int8) - 1
        h = H.local_fields(s)
        dE_vec = H.delta_energy_all(s, h=h)
        for i in range(model.n):
            # 1e-12 atol: both go through identical 2 s_i h_i arithmetic
            _assert_close(dE_vec[i], H.delta_energy(s, i, h=h), atol=1e-12)

# couplings must be symmetric with zero diagonal; trivial magnetization sanity check
def test_magnetization_and_zero_diag():
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
