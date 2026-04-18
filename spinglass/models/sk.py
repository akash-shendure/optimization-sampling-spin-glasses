# sherrington-kirkpatrick: dense mean-field glass on the complete graph
from .base import SpinModel
from ..couplings.sk import build_sk_couplings

# all-to-all couplings; gaussian draws use scale 1/sqrt(n) by default for thermodynamic limit
class SherringtonKirkpatrick(SpinModel):
    name = "SherringtonKirkpatrick"
    topology = "complete_graph"

    def __init__(self, n, scale=None):
        self.scale = scale
        J = build_sk_couplings(n, scale=scale)
        super().__init__(n=n, J=J)

    # params for logging / experiment manifests
    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "n": self.n,
            "scale": self.scale,
        }
