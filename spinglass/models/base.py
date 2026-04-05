# abstract spin model — holds n, coupling matrix J, rng, validation, and s sampling
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp

from ..utils.rng import make_rng

# common interface for ising, ea, sparse glass, sk — subclasses build J and call super
class SpinModel(ABC):
    name: str = "SpinModel"
    topology: str = "generic"

    def __init__(self, n, J, seed=None):
        self._n = int(n)
        self._J = J
        self._rng = make_rng(seed)
        self._validate()

    # number of spins
    @property
    def n(self):
        return self._n

    # coupling matrix (dense ndarray or scipy sparse)
    @property
    def J(self):
        return self._J

    # model-owned rng for reproducible state draws
    @property
    def rng(self):
        return self._rng

    # sparse couplings need different code paths in samplers/diagnostics
    @property
    def is_sparse(self):
        return sp.issparse(self._J)

    # avg nonzeros per row — proxy for graph connectivity
    def mean_degree(self):
        if self.is_sparse:
            return self._J.getnnz() / self._n
        return float(np.count_nonzero(self._J)) / self._n

    # uniform s in {-1, +1}^n; caller may pass an external rng
    def random_state(self, rng=None):
        r = self._rng if rng is None else make_rng(rng)
        return r.choice(np.array([-1, 1], dtype=np.int8), size=self._n)

    # checks any J builder must satisfy: square, symmetric, zero diagonal
    def _validate(self):
        assert self._J.shape == (self._n, self._n), "J shape mismatch"
        if self.is_sparse:
            diff = (self._J - self._J.T)
            assert abs(diff).sum() < 1e-10, "J not symmetric"  # tol for fp roundoff
        else:
            assert np.allclose(self._J, self._J.T), "J not symmetric"
        if self.is_sparse:
            assert np.all(self._J.diagonal() == 0), "J diagonal must be zero"
        else:
            assert np.all(np.diag(self._J) == 0), "J diagonal must be zero"

    # short tag with name, size, topology, mean degree — handy in logs
    def __repr__(self):
        return f"<{self.name} n={self._n} topology={self.topology} deg~{self.mean_degree():.2f}>"

    # subclasses return a dict of their defining params for run logs
    @abstractmethod
    def describe(self):
        ...
