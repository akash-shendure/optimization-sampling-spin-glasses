"""abstract base class for all spin models in the project."""
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng


# every model exposes: coupling matrix J, size n, topology tag, random spin state
# the Hamiltonian classes take a model and do the energy math; the model itself
# is just a container for the disorder realization + geometry metadata
class SpinModel(ABC):
    # subclasses must set these in __init__
    name: str = "SpinModel"
    topology: str = "generic"

    def __init__(self, n, J, seed=None):
        self._n = int(n)
        # store J as-is (sparse or dense); downstream code uses @ which works for both
        self._J = J
        self._rng = make_rng(seed)
        # cheap sanity checks — catch shape/symmetry bugs early
        self._validate()

    @property
    def n(self):
        return self._n

    @property
    def J(self):
        return self._J

    @property
    def rng(self):
        return self._rng

    # is the matrix representation sparse? useful for picking code paths later
    @property
    def is_sparse(self):
        return sp.issparse(self._J)

    # average number of nonzero neighbors per site — proxy for connectivity
    def mean_degree(self):
        if self.is_sparse:
            return self._J.getnnz() / self._n
        return float(np.count_nonzero(self._J)) / self._n

    # draw a uniform spin configuration in {-1,+1}^n
    def random_state(self, rng=None):
        r = self._rng if rng is None else make_rng(rng)
        return r.choice(np.array([-1, 1], dtype=np.int8), size=self._n)

    # shape + symmetry + zero-diagonal checks
    def _validate(self):
        assert self._J.shape == (self._n, self._n), "J shape mismatch"
        # symmetry check — skip for very large dense for speed, but do it here
        if self.is_sparse:
            diff = (self._J - self._J.T)
            assert abs(diff).sum() < 1e-10, "J not symmetric"
        else:
            assert np.allclose(self._J, self._J.T), "J not symmetric"
        # zero diagonal
        if self.is_sparse:
            assert np.all(self._J.diagonal() == 0), "J diagonal must be zero"
        else:
            assert np.all(np.diag(self._J) == 0), "J diagonal must be zero"

    # subclasses can override for custom repr
    def __repr__(self):
        return f"<{self.name} n={self._n} topology={self.topology} deg~{self.mean_degree():.2f}>"

    # hook: subclasses may override to expose geometric info (e.g. lattice shape)
    @abstractmethod
    def describe(self):
        ...
