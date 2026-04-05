# coupling-matrix generators — one submodule per topology
from .lattice import build_lattice_edges, ferromagnet_couplings, ea_couplings
from .sparse_graph import build_erdos_renyi_couplings
from .sk import build_sk_couplings
