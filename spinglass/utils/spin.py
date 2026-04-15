"""helpers for spin-state updates shared by discrete methods.

the hot loop of single-spin Metropolis / Gibbs / greedy / PT is
h <- h + 2 s_new_i * J[:, i]. pulling a column out of a CSR sparse matrix is
slow because CSR is row-major, so we cache a per-column (rows, data) pair
once at the start of the run and reuse it from there. for moderate n this
can be a 5-10x speedup on sparse models without touching the sampler logic."""
import numpy as np
import scipy.sparse as sp


def spin_column(J, i):
    """dense copy of column i of J. works for both sparse CSR and dense J."""
    col = J[:, i]
    if sp.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col, dtype=np.float64).ravel()


def update_local_fields(h, J, i, s_new_i):
    """in-place h += 2 * s_new_i * J[:, i]; legacy (slow) path without cache."""
    h += 2.0 * float(s_new_i) * spin_column(J, i)
    return h


class ColumnCache:
    """precomputed columns of J, keyed by column index.

    for sparse J we keep a pair of (row_indices, data) arrays per column —
    applying an update is then a single indexed add into h, O(k_i) per column
    with no CSR search. for dense J we keep the original matrix and slice
    into it on demand (already cheap)."""

    def __init__(self, J):
        self.n = int(J.shape[0])
        self.sparse = sp.issparse(J)
        if self.sparse:
            J_csc = J.tocsc()
            indptr = J_csc.indptr
            indices = J_csc.indices
            data = J_csc.data.astype(np.float64, copy=False)
            self._rows = []
            self._data = []
            for i in range(self.n):
                lo, hi = int(indptr[i]), int(indptr[i + 1])
                self._rows.append(np.ascontiguousarray(indices[lo:hi], dtype=np.int64))
                self._data.append(np.ascontiguousarray(data[lo:hi], dtype=np.float64))
            self._dense = None
        else:
            self._rows = None
            self._data = None
            self._dense = np.asarray(J, dtype=np.float64)

    def apply_flip(self, h, i, s_new_i):
        """in-place h += 2 * s_new_i * J[:, i] using the cached column."""
        scale = 2.0 * float(s_new_i)
        if self.sparse:
            rows = self._rows[i]
            data = self._data[i]
            if rows.size:
                h[rows] += scale * data
        else:
            h += scale * self._dense[:, i]
        return h


def update_local_fields_fast(h, cache, i, s_new_i):
    """same contract as update_local_fields but uses a precomputed ColumnCache.

    cache must have been built from the same J the Hamiltonian carries."""
    return cache.apply_flip(h, i, s_new_i)
