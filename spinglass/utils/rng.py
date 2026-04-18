"""RNG helper. every call returns a fresh, OS-randomized Generator."""
import numpy as np


def make_rng():
    return np.random.default_rng()
