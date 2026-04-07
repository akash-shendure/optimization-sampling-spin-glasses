"""helpers for spin-state updates shared by discrete methods."""
import numpy as np
import scipy.sparse as sp


def spin_column(J, i):
    col = J[:, i]
    if sp.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col, dtype=np.float64).ravel()


def update_local_fields(h, J, i, s_new_i):
    h += 2.0 * float(s_new_i) * spin_column(J, i)
    return h

