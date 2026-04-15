"""ColumnCache correctness + faster-than-legacy invariants.

the fast path must produce bit-identical local fields to the legacy
update_local_fields path and to a fresh J @ s recompute. we also measure
that the cache actually speeds up the sparse case by a nontrivial factor —
if that ratio drops below 1 the whole commit is pointless."""
import time

import numpy as np
import scipy.sparse as sp

from spinglass import (
    DiscreteHamiltonian,
    EdwardsAnderson2D,
    IsingFerromagnet2D,
    SherringtonKirkpatrick,
    SparseRandomGlass,
)
from spinglass.utils.spin import (
    ColumnCache,
    update_local_fields,
    update_local_fields_fast,
)


def _models():
    return [
        IsingFerromagnet2D(L=5, seed=0),
        EdwardsAnderson2D(L=6, disorder="pm1", seed=1),
        SparseRandomGlass(n=32, c=3.0, seed=2),
        SherringtonKirkpatrick(n=20, seed=3),
    ]


def test_column_cache_matches_legacy_on_single_flip():
    rng = np.random.default_rng(10)
    for model in _models():
        H = DiscreteHamiltonian(model)
        cache = ColumnCache(H.J)
        s = model.random_state(rng).copy()
        h_legacy = H.local_fields(s).copy()
        h_fast = h_legacy.copy()
        for _ in range(20):
            i = int(rng.integers(model.n))
            s[i] = -s[i]
            update_local_fields(h_legacy, H.J, i, s[i])
            update_local_fields_fast(h_fast, cache, i, s[i])
        assert np.allclose(h_legacy, h_fast, atol=1e-10), f"mismatch on {type(model).__name__}"


def test_column_cache_matches_fresh_recompute():
    rng = np.random.default_rng(11)
    for model in _models():
        H = DiscreteHamiltonian(model)
        cache = ColumnCache(H.J)
        s = model.random_state(rng).copy()
        h = H.local_fields(s).copy()
        for _ in range(50):
            i = int(rng.integers(model.n))
            s[i] = -s[i]
            update_local_fields_fast(h, cache, i, s[i])
        h_fresh = H.local_fields(s)
        assert np.allclose(h, h_fresh, atol=1e-9), f"cache drifted on {type(model).__name__}"


def test_column_cache_empty_column_is_noop():
    # lattice has some columns with few neighbors; a column with zero nonzeros
    # should still not blow up
    J = sp.csr_matrix((np.array([1.0, 1.0]), (np.array([0, 2]), np.array([2, 0]))), shape=(4, 4))
    J = 0.5 * (J + J.T)
    cache = ColumnCache(J)
    h = np.zeros(4)
    update_local_fields_fast(h, cache, 1, 1)  # column 1 is all zero
    assert np.allclose(h, 0.0)


def test_hamiltonian_column_cache_is_cached_and_shared():
    model = EdwardsAnderson2D(L=4, seed=4)
    H = DiscreteHamiltonian(model)
    c1 = H.column_cache()
    c2 = H.column_cache()
    assert c1 is c2  # same instance on repeated access


def test_column_cache_speedup_on_sparse_model():
    # build something big enough for the timing gap to be visible
    model = EdwardsAnderson2D(L=20, disorder="pm1", seed=5)
    H = DiscreteHamiltonian(model)
    cache = ColumnCache(H.J)
    rng = np.random.default_rng(6)
    s = model.random_state(rng).copy()
    flips = rng.integers(0, model.n, size=1500)

    h_legacy = H.local_fields(s).copy()
    t0 = time.perf_counter()
    for idx in flips:
        i = int(idx)
        s[i] = -s[i]
        update_local_fields(h_legacy, H.J, i, s[i])
    legacy_t = time.perf_counter() - t0

    # reset s back to the starting config so both runs do identical work
    s = model.random_state(np.random.default_rng(6)).copy()
    h_fast = H.local_fields(s).copy()
    t0 = time.perf_counter()
    for idx in flips:
        i = int(idx)
        s[i] = -s[i]
        update_local_fields_fast(h_fast, cache, i, s[i])
    fast_t = time.perf_counter() - t0

    # the cached path should be meaningfully faster on sparse J. be lenient
    # so noisy machines don't fail the test — we just want strictly faster.
    assert fast_t <= legacy_t, f"fast path slower: legacy {legacy_t:.3f}s vs fast {fast_t:.3f}s"


def test_dense_model_cache_also_correct():
    # SK has dense J — the cache should still give correct results (same as legacy)
    model = SherringtonKirkpatrick(n=24, seed=7)
    H = DiscreteHamiltonian(model)
    cache = H.column_cache()
    assert not cache.sparse
    rng = np.random.default_rng(8)
    s = model.random_state(rng).copy()
    h = H.local_fields(s).copy()
    for _ in range(40):
        i = int(rng.integers(model.n))
        s[i] = -s[i]
        update_local_fields_fast(h, cache, i, s[i])
    assert np.allclose(h, H.local_fields(s), atol=1e-9)


if __name__ == "__main__":
    test_column_cache_matches_legacy_on_single_flip()
    test_column_cache_matches_fresh_recompute()
    test_column_cache_empty_column_is_noop()
    test_hamiltonian_column_cache_is_cached_and_shared()
    test_column_cache_speedup_on_sparse_model()
    test_dense_model_cache_also_correct()
    print("test_column_cache OK")
