# sparse random glass: erdos-renyi graph with random ±1 or gaussian bonds
from .base import SpinModel
from ..couplings.sparse_graph import build_erdos_renyi_couplings

# average degree c controls connectivity; bridges lattice EA and dense SK
class SparseRandomGlass(SpinModel):
    name = "SparseRandomGlass"
    topology = "erdos_renyi"

    # c = expected degree; scale rescales bond magnitudes
    def __init__(self, n, c=3.0, disorder="gaussian", scale=1.0):
        self.c = float(c)
        self.disorder = disorder
        self.scale = float(scale)
        J = build_erdos_renyi_couplings(
            n, c=self.c, disorder=self.disorder, scale=self.scale
        )
        super().__init__(n=n, J=J)

    # params for logging / experiment manifests
    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "n": self.n,
            "c": self.c,
            "disorder": self.disorder,
            "scale": self.scale,
        }
