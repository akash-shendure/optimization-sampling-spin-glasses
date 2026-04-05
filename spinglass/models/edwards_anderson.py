# 2d edwards-anderson spin glass on an LxL lattice with random ±1 or gaussian bonds
from .base import SpinModel
from ..couplings.lattice import ea_couplings

# frustrated nearest-neighbor lattice — couplings have random signs
class EdwardsAnderson2D(SpinModel):
    name = "EdwardsAnderson2D"
    topology = "lattice_2d"

    def __init__(self, L, disorder="pm1", scale=1.0, periodic=True, seed=None):
        self.L = int(L)
        self.disorder = disorder
        self.scale = float(scale)
        self.periodic = bool(periodic)
        J = ea_couplings(
            self.L,
            disorder=self.disorder,
            scale=self.scale,
            seed=seed,
            periodic=self.periodic,
        )
        super().__init__(n=self.L * self.L, J=J, seed=seed)

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
