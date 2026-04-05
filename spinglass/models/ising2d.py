"""2D ferromagnetic Ising model on a periodic square lattice."""
from .base import SpinModel
from ..couplings.lattice import ferromagnet_couplings


class IsingFerromagnet2D(SpinModel):
    name = "IsingFerromagnet2D"
    topology = "lattice_2d"

    # L is side length; n = L*L
    def __init__(self, L, J0=1.0, periodic=True, seed=None):
        self.L = int(L)
        self.J0 = float(J0)
        self.periodic = bool(periodic)
        J = ferromagnet_couplings(self.L, J0=self.J0, periodic=self.periodic)
        super().__init__(n=self.L * self.L, J=J, seed=seed)

    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "L": self.L,
            "n": self.n,
            "J0": self.J0,
            "periodic": self.periodic,
        }
