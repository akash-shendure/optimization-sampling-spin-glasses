"""matplotlib helpers for traces, diagnostics, and difficulty curves."""
from .diagnostics import (
    plot_acf,
    plot_pair,
    plot_pair_matrix,
    plot_rank_histogram,
    plot_trace,
)
from .difficulty import plot_difficulty_curve, plot_grouped_metric
from .traceplots import plot_optimizer_trace, plot_sampler_trace

__all__ = [
    "plot_acf",
    "plot_pair",
    "plot_pair_matrix",
    "plot_rank_histogram",
    "plot_trace",
    "plot_difficulty_curve",
    "plot_grouped_metric",
    "plot_optimizer_trace",
    "plot_sampler_trace",
]

