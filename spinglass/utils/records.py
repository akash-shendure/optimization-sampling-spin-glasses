"""small helpers for run traces and result summaries."""
import time
import numpy as np


def now():
    return time.perf_counter()


def init_trace():
    return {}


def append_trace(trace, **kwargs):
    for key, value in kwargs.items():
        trace.setdefault(key, []).append(value)


def finalize_trace(trace):
    out = {}
    for key, values in trace.items():
        out[key] = np.asarray(values)
    return out

