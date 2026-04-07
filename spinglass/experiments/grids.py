# tiny grid utilities for sweep configs — cartesian product + dict merge
from itertools import product

# yield one dict per cartesian-product point; scalars are treated as length-1 lists
def parameter_grid(grid_dict):
    # empty grid yields exactly one empty config (used for "no sweep" runs)
    if not grid_dict:
        yield {}
        return
    keys = list(grid_dict.keys())
    values = []
    # normalize each value to a list so product() works uniformly
    for key in keys:
        value = grid_dict[key]
        if isinstance(value, (list, tuple)):
            values.append(list(value))
        else:
            values.append([value])
    for combo in product(*values):
        yield dict(zip(keys, combo))

# shallow merge — later dicts override earlier ones
def merge_dicts(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out
