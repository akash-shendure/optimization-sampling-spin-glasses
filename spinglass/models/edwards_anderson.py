"""2D Edwards-Anderson spin glass: lattice edges, random signs."""
from .base import SpinModel
from ..couplings.lattice import ea_couplings


class EdwardsAnderson2D(SpinModel):
    name = "EdwardsAnderson2D"
    topology = "lattice_2d"

    def __init__(self, L, disorder="pm1", scale=1.0, periodic=True, seed=None):
        self.L = int(L)
        self.disorder = disorder
        self.scale = float(scale)
        self.periodic = bool(periodic)
        # seed the disorder realization from the same seed the base will use
        J = ea_couplings(
            self.L,
            disorder=self.disorder,
            scale=self.scale,
            seed=seed,
            periodic=self.periodic,
        )
        super().__init__(n=self.L * self.L, J=J, seed=seed)

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
