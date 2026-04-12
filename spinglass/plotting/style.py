# publication-style rcParams plus the per-panel color palette used across the project
import matplotlib as mpl
import matplotlib.pyplot as plt

# stable color per algorithm family — keeps figures consistent across notebooks
PANEL_COLORS = {
    "discrete_sampling": "#1f77b4",
    "discrete_optimization": "#d62728",
    "relaxed_sampling": "#2ca02c",
    "relaxed_optimization": "#ff7f0e",
    "parallel_tempering": "#9467bd",
    "other": "#7f7f7f",
}

# tuned for two-column paper figures: small fonts, no top/right spines, faint grid
_PUB_RCPARAMS = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,    # why: cleaner look, draws the eye to data not box
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
    "legend.frameon": False,    # why: frame competes with grid lines
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
}

# rcparam snapshot stack so set/reset can be nested safely
_stack = []

# push current rcparams and install the publication style
def set_publication_style():
    _stack.append(dict(mpl.rcParams))
    mpl.rcParams.update(_PUB_RCPARAMS)

# pop the last snapshot; fall back to matplotlib defaults if the stack is empty
def reset_style():
    if not _stack:
        mpl.rcdefaults()
        return
    prev = _stack.pop()
    mpl.rcParams.update(prev)

# look up a family color, falling back to the neutral "other" entry
def panel_color(name):
    return PANEL_COLORS.get(name, PANEL_COLORS["other"])

# save fig at print dpi and close it — important inside loops to free memory
def save_figure(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight", dpi=200)
    plt.close(fig)
