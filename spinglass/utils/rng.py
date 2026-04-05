# single entry point for rng creation; numpy Generator seeded from os entropy
import numpy as np

def make_rng(seed=None):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)

def spawn_rng(parent, n=1):
    ss = np.random.SeedSequence(parent.integers(0, 2**63 - 1, dtype=np.int64))
    children = ss.spawn(n)
    return [np.random.default_rng(c) for c in children]
