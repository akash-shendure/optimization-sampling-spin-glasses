"""helpers for running algorithm sweeps and saving results."""
from .benchmarks import collect_chain_traces, summarize_optimization_table, summarize_sampling_table
from .builders import build_algorithm, build_hamiltonian, build_model
from .grids import merge_dicts, parameter_grid
from .io import ensure_dir, save_json, save_npz
from .runner import flatten_record, run_grid, run_single

__all__ = [
    "collect_chain_traces",
    "summarize_optimization_table",
    "summarize_sampling_table",
    "build_algorithm",
    "build_hamiltonian",
    "build_model",
    "merge_dicts",
    "parameter_grid",
    "ensure_dir",
    "save_json",
    "save_npz",
    "flatten_record",
    "run_single",
    "run_grid",
]
