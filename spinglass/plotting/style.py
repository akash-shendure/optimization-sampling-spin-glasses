"""publication-ready matplotlib styling.

intentionally thin: we set rcParams rather than mandating a wrapper, so any
plotting helper in this package or in a notebook picks up the same defaults
after one call to set_publication_style(). invert/undo is a two-call cycle —
push rcParams, pop rcParams — to avoid leaking into unrelated figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt


# the six colors used across benchmark plots; chosen for decent contrast on
# both screen and greyscale print. order: discrete-sampling, discrete-opt,
# relaxed-sampling, relaxed-opt, PT, fallback
PANEL_COLORS = {
    "discrete_sampling": "#1f77b4",
    "discrete_optimization": "#d62728",
    "relaxed_sampling": "#2ca02c",
    "relaxed_optimization": "#ff7f0e",
    "parallel_tempering": "#9467bd",
    "other": "#7f7f7f",
}


_PUB_RCPARAMS = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "#cccccc",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
}


_stack = []


def set_publication_style():
    """push the publication rcParams onto matplotlib.

    call once at the top of a notebook or script. safe to call repeatedly —
    each call stacks another snapshot, and reset_style() pops one."""
    _stack.append(dict(mpl.rcParams))
    mpl.rcParams.update(_PUB_RCPARAMS)


def reset_style():
    """pop the most recent rcParams snapshot saved by set_publication_style."""
    if not _stack:
        mpl.rcdefaults()
        return
    prev = _stack.pop()
    mpl.rcParams.update(prev)


def panel_color(name):
    return PANEL_COLORS.get(name, PANEL_COLORS["other"])


def save_figure(fig, path, tight=True):
    """save a figure with the publication defaults regardless of current rc."""
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    plt.close(fig)
