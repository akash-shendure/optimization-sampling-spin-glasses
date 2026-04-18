"""incremental local-field update invariants.

single-flip Metropolis / Gibbs / PT rely on maintaining h = J @ s incrementally
instead of recomputing J @ s after every accepted flip. this test verifies
update_local_fields stays consistent with a fresh recompute over many flips,
both for sparse CSR J (lattice / Erdos-Renyi) and dense J (SK)."""
import numpy as np

from spinglass import DiscreteHamiltonian, EdwardsAnderson2D, SherringtonKirkpatrick, SparseRandomGlass
from spinglass.utils.spin import spin_column, update_local_fields


def _simulate_flips(model, n_flips):
    H = DiscreteHamiltonian(model)
    rng = np.random.default_rng()
    s = model.random_state(rng).copy()
    h = H.local_fields(s).copy()
    for _ in range(n_flips):
        i = int(rng.integers(model.n))
        s[i] = -s[i]
        update_local_fields(h, H.J, i, s[i])
    return s, h, H.local_fields(s)


def test_incremental_fields_match_fresh_recompute_sparse():
    model = EdwardsAnderson2D(L=6, disorder="pm1")
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=50)
    assert np.allclose(h_inc, h_fresh, atol=1e-10)


def test_incremental_fields_match_fresh_recompute_er():
    model = SparseRandomGlass(n=40, c=4.0)
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=80)
    assert np.allclose(h_inc, h_fresh, atol=1e-9)


def test_incremental_fields_match_fresh_recompute_sk():
    model = SherringtonKirkpatrick(n=30)
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=100)
    assert np.allclose(h_inc, h_fresh, atol=1e-9)


def test_spin_column_matches_dense_column():
    rng = np.random.default_rng()
    for model in (EdwardsAnderson2D(L=4), SherringtonKirkpatrick(n=16)):
        J_dense = model.J.toarray() if hasattr(model.J, "toarray") else np.asarray(model.J)
        for _ in range(3):
            i = int(rng.integers(model.n))
            col = spin_column(model.J, i)
            assert col.shape == (model.n,)
            assert np.allclose(col, J_dense[:, i], atol=1e-12)


def test_flip_twice_restores_local_fields():
    # flipping the same site twice should leave h unchanged — direct consistency check
    model = EdwardsAnderson2D(L=5)
    H = DiscreteHamiltonian(model)
    rng = np.random.default_rng()
    s = model.random_state(rng).copy()
    h0 = H.local_fields(s).copy()
    for i in (0, 3, 7, 11):
        h = h0.copy()
        s_local = s.copy()
        s_local[i] = -s_local[i]
        update_local_fields(h, H.J, i, s_local[i])
        s_local[i] = -s_local[i]
        update_local_fields(h, H.J, i, s_local[i])
        assert np.allclose(h, h0, atol=1e-12)


if __name__ == "__main__":
    test_incremental_fields_match_fresh_recompute_sparse()
    test_incremental_fields_match_fresh_recompute_er()
    test_incremental_fields_match_fresh_recompute_sk()
    test_spin_column_matches_dense_column()
    test_flip_twice_restores_local_fields()
    print("test_incremental_fields OK")
