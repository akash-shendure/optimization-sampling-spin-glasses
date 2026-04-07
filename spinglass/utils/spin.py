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
