"""sparse random graph Ising glass: ER topology, random couplings."""
from .base import SpinModel
from ..couplings.sparse_graph import build_erdos_renyi_couplings


class SparseRandomGlass(SpinModel):
    name = "SparseRandomGlass"
    topology = "erdos_renyi"

    # c is target mean degree (so p = c/(n-1))
    def __init__(self, n, c=3.0, disorder="gaussian", scale=1.0):
        self.c = float(c)
        self.disorder = disorder
        self.scale = float(scale)
        J = build_erdos_renyi_couplings(
            n, c=self.c, disorder=self.disorder, scale=self.scale
        )
        super().__init__(n=n, J=J)

    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "n": self.n,
            "c": self.c,
            "disorder": self.disorder,
            "scale": self.scale,
        }
