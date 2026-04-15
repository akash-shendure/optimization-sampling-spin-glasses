# local-field updates for single-spin flips; O(k_i) via cached nonzero columns
import numpy as np
import scipy.sparse as sp

# extract column i of J as a dense 1d float64 array (handles sparse and dense J)
def spin_column(J, i):
    col = J[:, i]
    if sp.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col, dtype=np.float64).ravel()

# h_j += 2 s_new_i J_{ji}  after flipping spin i; correct because delta s_i = 2 s_new_i
def update_local_fields(h, J, i, s_new_i):
    h += 2.0 * float(s_new_i) * spin_column(J, i)
    return h

# precomputes nonzero rows of each column of J once per disorder realization
# so per-flip updates touch only O(k_i) entries instead of O(n)
class ColumnCache:

    # split sparse J into per-column (row indices, data) buffers; keep dense J as-is
    def __init__(self, J):
        self.n = int(J.shape[0])
        self.sparse = sp.issparse(J)
        if self.sparse:
            # csc gives O(1) column slices; copy indices/data into contiguous arrays
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
            # dense path: just hold a float64 view of J
            self._rows = None
            self._data = None
            self._dense = np.asarray(J, dtype=np.float64)

    # apply h += 2 s_new_i J[:, i] touching only nonzero rows for sparse J
    def apply_flip(self, h, i, s_new_i):
        scale = 2.0 * float(s_new_i)
        if self.sparse:
            rows = self._rows[i]
            data = self._data[i]
            # skip if column i has no nonzeros (isolated spin)
            if rows.size:
                h[rows] += scale * data
        else:
            h += scale * self._dense[:, i]
        return h

# thin wrapper so callers can swap update_local_fields -> update_local_fields_fast
def update_local_fields_fast(h, cache, i, s_new_i):
    return cache.apply_flip(h, i, s_new_i)
