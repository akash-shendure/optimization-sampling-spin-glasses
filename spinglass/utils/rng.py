# single entry point for rng creation; numpy Generator seeded from os entropy
import numpy as np

# fresh Generator per call -> independent randomness across runs/chains
def make_rng():
    return np.random.default_rng()
