# update_local_fields_fast must keep h in sync with J s after each single-spin flip
import numpy as np

from spinglass import DiscreteHamiltonian, EdwardsAnderson2D, SherringtonKirkpatrick, SparseRandomGlass
from spinglass.utils.spin import spin_column, update_local_fields

# helper: flip n_flips random spins incrementally, return final s, incremental h, fresh h
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

# 2D EA (sparse pm1): 50 flips, incremental h must equal fresh recompute
def test_incremental_fields_match_fresh_recompute_sparse():
    model = EdwardsAnderson2D(L=6, disorder="pm1")
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=50)
    assert np.allclose(h_inc, h_fresh, atol=1e-10)

# Erdos-Renyi sparse glass: 80 flips, looser atol since gaussian couplings are non-integer
def test_incremental_fields_match_fresh_recompute_er():
    model = SparseRandomGlass(n=40, c=4.0)
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=80)
    assert np.allclose(h_inc, h_fresh, atol=1e-9)

# SK dense: 100 flips, atol=1e-9 because dense gaussian J amplifies rounding a touch
def test_incremental_fields_match_fresh_recompute_sk():
    model = SherringtonKirkpatrick(n=30)
    s, h_inc, h_fresh = _simulate_flips(model, n_flips=100)
    assert np.allclose(h_inc, h_fresh, atol=1e-9)

# spin_column(J, i) must return the i-th column densely, matching toarray()
def test_spin_column_matches_dense_column():
    rng = np.random.default_rng()
    for model in (EdwardsAnderson2D(L=4), SherringtonKirkpatrick(n=16)):
        J_dense = model.J.toarray() if hasattr(model.J, "toarray") else np.asarray(model.J)
        for _ in range(3):
            i = int(rng.integers(model.n))
            col = spin_column(model.J, i)
            assert col.shape == (model.n,)
            # 1e-12 atol: integer / direct float reads, no accumulated error
            assert np.allclose(col, J_dense[:, i], atol=1e-12)

# flipping the same site twice must be a perfect identity on h (involution)
def test_flip_twice_restores_local_fields():
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
        # 1e-12 atol: same delta added then subtracted, no drift expected
        assert np.allclose(h, h0, atol=1e-12)

if __name__ == "__main__":
    test_incremental_fields_match_fresh_recompute_sparse()
    test_incremental_fields_match_fresh_recompute_er()
    test_incremental_fields_match_fresh_recompute_sk()
    test_spin_column_matches_dense_column()
    test_flip_twice_restores_local_fields()
    print("test_incremental_fields OK")
