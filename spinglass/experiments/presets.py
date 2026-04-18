# named scaling-study configurations — model + grid + beta sweep for the CLI
from ..models.edwards_anderson import EdwardsAnderson2D
from ..models.ising2d import IsingFerromagnet2D
from ..models.sk import SherringtonKirkpatrick
from ..models.sparse_glass import SparseRandomGlass
from .budget import sweeps

# each preset is consumed by studies.canonical_study via the CLI
# budgets are in sweeps so discrete and relaxed see equal Hamiltonian work
PRESETS = {
    # 2d ferromagnet — beta sweep brackets T_c~0.44, no disorder needed
    "ferro_scaling": {
        "model_class": IsingFerromagnet2D,
        "model_kwargs": {"L": [8, 12, 16]},
        "betas": [0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.2],
        "n_disorders": 1,
        "n_chains": 4,
        "n_restarts": 4,
        "budget": sweeps(100),
        "description": "2D ferromagnet L in {8,12,16}, beta sweep across T_c~0.44",
    },
    # 2d edwards-anderson glass — average over 4 disorder draws
    "ea_scaling": {
        "model_class": EdwardsAnderson2D,
        "model_kwargs": {"L": [8, 12], "disorder": ["pm1"]},
        "betas": [0.3, 0.6, 1.0, 1.5, 2.0, 3.0],
        "n_disorders": 4,
        "n_chains": 4,
        "n_restarts": 4,
        "budget": sweeps(150),
        "description": "Edwards-Anderson pm1 glass L in {8,12}, 4 disorder realizations",
    },
    # erdos-renyi sparse random graph with gaussian couplings
    "er_glass": {
        "model_class": SparseRandomGlass,
        "model_kwargs": {"n": [64, 128], "c": [3.0], "disorder": ["gaussian"]},
        "betas": [0.3, 0.8, 1.5, 2.5, 4.0],
        "n_disorders": 3,
        "n_chains": 4,
        "n_restarts": 4,
        "budget": sweeps(150),
        "description": "Erdos-Renyi Gaussian glass n in {64,128}, mean degree 3",
    },
    # sherrington-kirkpatrick — mean-field reference glass
    "sk": {
        "model_class": SherringtonKirkpatrick,
        "model_kwargs": {"n": [48, 64]},
        "betas": [0.3, 0.6, 1.0, 1.5, 2.0],
        "n_disorders": 3,
        "n_chains": 4,
        "n_restarts": 4,
        "budget": sweeps(150),
        "description": "Sherrington-Kirkpatrick dense mean-field n in {48,64}",
    },
    # tiny preset for tests — finishes in seconds
    "smoke": {
        "model_class": IsingFerromagnet2D,
        "model_kwargs": {"L": [6]},
        "betas": [0.3, 0.8],
        "n_disorders": 1,
        "n_chains": 2,
        "n_restarts": 2,
        "budget": sweeps(30),
        "description": "tiny smoke preset — use for CI and debugging",
    },
}

# sorted preset names for help text / CLI tab completion
def list_presets():
    return sorted(PRESETS.keys())

# fetch a copy so callers can mutate without polluting the registry
def get_preset(name):
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(list_presets())}"
        )
    return dict(PRESETS[name])
