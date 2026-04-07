# trace dict bookkeeping for samplers/optimizers plus a monotonic clock
import time
import numpy as np

# monotonic high-resolution clock; used for runtime_sec measurements
def now():
    return time.perf_counter()

# fresh empty trace; samplers append per-step scalars/arrays into this dict
def init_trace():
    return {}

# push one value per keyword arg onto its named list (creating the list on first use)
def append_trace(trace, **kwargs):
    for key, value in kwargs.items():
        trace.setdefault(key, []).append(value)

# convert lists-of-values to numpy arrays for downstream analysis/plotting
def finalize_trace(trace):
    out = {}
    for key, values in trace.items():
        out[key] = np.asarray(values)
    return out
