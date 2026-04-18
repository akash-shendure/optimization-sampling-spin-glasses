# 2d edwards-anderson spin glass on an LxL lattice with random ±1 or gaussian bonds
from .base import SpinModel
from ..couplings.lattice import ea_couplings

# frustrated nearest-neighbor lattice — couplings have random signs
class EdwardsAnderson2D(SpinModel):
    name = "EdwardsAnderson2D"
    topology = "lattice_2d"

    # disorder='pm1' for ±1 bonds, 'gaussian' for N(0, scale^2); periodic toggles wrap
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
        super().__init__(n=self.L * self.L, J=J)  # n = L^2 sites

    # params for logging / experiment manifests
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
