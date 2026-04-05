"""RNG helpers. keeps seeding consistent across the project."""
import numpy as np


# accept int, None, or an existing Generator — always return a Generator
def make_rng(seed=None):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# derive an independent child stream from a parent (for per-instance disorder)
def spawn_rng(parent, n=1):
    # SeedSequence spawning gives statistically independent streams
    ss = np.random.SeedSequence(parent.integers(0, 2**63 - 1, dtype=np.int64))
    children = ss.spawn(n)
    return [np.random.default_rng(c) for c in children]
