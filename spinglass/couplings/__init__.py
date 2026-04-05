# public surface of the couplings package — exposes J-matrix builders for each model family
from .lattice import build_lattice_edges, ferromagnet_couplings, ea_couplings
from .sparse_graph import build_erdos_renyi_couplings
from .sk import build_sk_couplings
