"""Sherrington-Kirkpatrick dense mean-field spin glass."""
from .base import SpinModel
from ..couplings.sk import build_sk_couplings


class SherringtonKirkpatrick(SpinModel):
    name = "SherringtonKirkpatrick"
    topology = "complete_graph"

    # scale=None defaults to 1/sqrt(n) (standard SK normalization)
    def __init__(self, n, scale=None, seed=None):
        self.scale = scale
        J = build_sk_couplings(n, scale=scale, seed=seed)
        super().__init__(n=n, J=J, seed=seed)

    def describe(self):
        return {
            "name": self.name,
            "topology": self.topology,
            "n": self.n,
            "scale": self.scale,
        }
