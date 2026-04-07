"""simple cartesian-product helpers for experiment sweeps."""
from itertools import product


def parameter_grid(grid_dict):
    if not grid_dict:
        yield {}
        return
    keys = list(grid_dict.keys())
    values = []
    for key in keys:
        value = grid_dict[key]
        if isinstance(value, (list, tuple)):
            values.append(list(value))
        else:
            values.append([value])
    for combo in product(*values):
        yield dict(zip(keys, combo))


def merge_dicts(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out

