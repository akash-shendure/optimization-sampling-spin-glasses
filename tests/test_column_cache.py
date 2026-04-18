# ColumnCache precomputes columns of J; verify integrity + speedup vs touching J directly
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

# a mixed bag of sparse and dense models so the cache is exercised against both
def _models():
    return [
        IsingFerromagnet2D(L=5),
        EdwardsAnderson2D(L=6, disorder="pm1"),
        SparseRandomGlass(n=32, c=3.0),
        SherringtonKirkpatrick(n=20),
    ]

# fast path with ColumnCache must agree with the legacy J-slicing path after each flip
def test_column_cache_matches_legacy_on_single_flip():
    rng = np.random.default_rng()
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
        # 1e-10 atol because both paths do the same algebra in float64
        assert np.allclose(h_legacy, h_fast, atol=1e-10), f"mismatch on {type(model).__name__}"

# incremental updates must not drift away from a fresh recomputation after many flips
def test_column_cache_matches_fresh_recompute():
    rng = np.random.default_rng()
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
        # 1e-9 atol: 50 flips of accumulation, still float64 safe
        assert np.allclose(h, h_fresh, atol=1e-9), f"cache drifted on {type(model).__name__}"

# flipping a spin whose column is empty must leave h untouched
def test_column_cache_empty_column_is_noop():
    J = sp.csr_matrix((np.array([1.0, 1.0]), (np.array([0, 2]), np.array([2, 0]))), shape=(4, 4))
    J = 0.5 * (J + J.T)
    cache = ColumnCache(J)
    h = np.zeros(4)
    update_local_fields_fast(h, cache, 1, 1)
    assert np.allclose(h, 0.0)

# DiscreteHamiltonian.column_cache() should memoize, not rebuild on each call
def test_hamiltonian_column_cache_is_cached_and_shared():
    model = EdwardsAnderson2D(L=4)
    H = DiscreteHamiltonian(model)
    c1 = H.column_cache()
    c2 = H.column_cache()
    # identity, not equality: same object returned  # why: avoid silent rebuild cost
    assert c1 is c2

# sparse-friendly model: fast path should be at least as quick as legacy J slicing
def test_column_cache_speedup_on_sparse_model():
    model = EdwardsAnderson2D(L=20, disorder="pm1")
    H = DiscreteHamiltonian(model)
    cache = ColumnCache(H.J)
    rng = np.random.default_rng()
    s0 = model.random_state(rng).copy()
    flips = rng.integers(0, model.n, size=1500)

    s = s0.copy()
    h_legacy = H.local_fields(s).copy()
    t0 = time.perf_counter()
    for idx in flips:
        i = int(idx)
        s[i] = -s[i]
        update_local_fields(h_legacy, H.J, i, s[i])
    legacy_t = time.perf_counter() - t0

    s = s0.copy()
    h_fast = H.local_fields(s).copy()
    t0 = time.perf_counter()
    for idx in flips:
        i = int(idx)
        s[i] = -s[i]
        update_local_fields_fast(h_fast, cache, i, s[i])
    fast_t = time.perf_counter() - t0

    # weak inequality, not strict: noisy CI machines occasionally tie
    assert fast_t <= legacy_t, f"fast path slower: legacy {legacy_t:.3f}s vs fast {fast_t:.3f}s"

# dense (SK) model: cache flag should report dense, and values still match
def test_dense_model_cache_also_correct():
    model = SherringtonKirkpatrick(n=24)
    H = DiscreteHamiltonian(model)
    cache = H.column_cache()
    assert not cache.sparse
    rng = np.random.default_rng()
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
