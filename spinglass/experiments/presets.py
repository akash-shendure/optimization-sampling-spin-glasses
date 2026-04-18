"""preset configurations for the canonical benchmark studies.

each preset defines a model class, a model kwargs grid, a beta grid, disorder
count, and a matched compute budget. the CLI and scripts/run_canonical_study
both read from this registry so scaling sweeps are a one-flag invocation:

    python -m spinglass canonical --preset ferro_scaling
    python -m spinglass canonical --preset ea_scaling
    python -m spinglass canonical --preset er_glass
    python -m spinglass canonical --preset sk

presets intentionally stay small in this repo — they're the "first benchmark
study" shape from the proposal. anything larger is a one-off notebook job."""
from ..models.edwards_anderson import EdwardsAnderson2D
from ..models.ising2d import IsingFerromagnet2D
from ..models.sk import SherringtonKirkpatrick
from ..models.sparse_glass import SparseRandomGlass
from .budget import sweeps


PRESETS = {
    "ferro_scaling": {
        "model_class": IsingFerromagnet2D,
        "model_kwargs": {"L": [8, 12, 16]},
        "betas": [0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.2],
        "n_disorders": 1,  # no disorder — one realization per L
        "n_chains": 4,
        "n_restarts": 4,
        "budget": sweeps(100),
        "description": "2D ferromagnet L in {8,12,16}, beta sweep across T_c~0.44",
    },
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
    "smoke": {
        # tiny preset for CI / quick sanity — runs in a few seconds
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


def list_presets():
    """list available preset names, sorted."""
    return sorted(PRESETS.keys())


def get_preset(name):
    """look up a preset by name, raising a clear error on miss."""
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(list_presets())}"
        )
    # shallow copy so callers can't mutate the module-level dict
    return dict(PRESETS[name])
