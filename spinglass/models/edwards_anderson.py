"""2D Edwards-Anderson spin glass: lattice edges, random signs."""
from .base import SpinModel
from ..couplings.lattice import ea_couplings


class EdwardsAnderson2D(SpinModel):
    name = "EdwardsAnderson2D"
    topology = "lattice_2d"

    def __init__(self, L, disorder="pm1", scale=1.0, periodic=True):
        self.L = int(L)
        self.disorder = disorder
        self.scale = float(scale)
        self.periodic = bool(periodic)
        J = ea_couplings(
            self.L,
            disorder=self.disorder,
            scale=self.scale,
            periodic=self.periodic,
        )
        super().__init__(n=self.L * self.L, J=J)

    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "L": self.L,
            "n": self.n,
            "disorder": self.disorder,
            "scale": self.scale,
            "periodic": self.periodic,
        }
