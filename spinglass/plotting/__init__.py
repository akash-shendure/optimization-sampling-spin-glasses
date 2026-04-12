"""matplotlib helpers for traces, diagnostics, and difficulty curves."""
from .diagnostics import (
    plot_acf,
    plot_pair,
    plot_pair_matrix,
    plot_rank_histogram,
    plot_trace,
)
from .difficulty import plot_difficulty_curve, plot_grouped_metric
from .overlap import (
    plot_mean_abs_q_curve,
    plot_overlap_histogram,
    plot_overlap_histograms_by_beta,
    plot_overlap_vs_energy,
)
from .style import PANEL_COLORS, panel_color, reset_style, save_figure, set_publication_style
from .traceplots import plot_optimizer_trace, plot_sampler_trace

__all__ = [
    "plot_acf",
    "plot_pair",
    "plot_pair_matrix",
    "plot_rank_histogram",
    "plot_trace",
    "plot_difficulty_curve",
    "plot_grouped_metric",
    "plot_mean_abs_q_curve",
    "plot_overlap_histogram",
    "plot_overlap_histograms_by_beta",
    "plot_overlap_vs_energy",
    "plot_optimizer_trace",
    "plot_sampler_trace",
    "PANEL_COLORS",
    "panel_color",
    "reset_style",
    "save_figure",
    "set_publication_style",
]
