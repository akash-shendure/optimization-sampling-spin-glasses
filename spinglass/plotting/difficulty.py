# difficulty-curve plots: metric (energy, success, mixing error) vs beta or another sweep axis
import numpy as np
import matplotlib.pyplot as plt

# generic grouped line plot over a list-of-dict benchmark table
# rows: list of dicts; x/y are column keys; group is an optional series key
def plot_grouped_metric(rows, x, y, group=None, ax=None, title=None, marker="o"):
    if not rows:
        raise ValueError("rows must be non-empty")
    ax = _get_ax(ax)
    if group is None:
        # single series: pull (x, y) pairs and sort by x so the line is monotone
        xs = np.asarray([row[x] for row in rows], dtype=np.float64)
        ys = np.asarray([row[y] for row in rows], dtype=np.float64)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=marker, lw=1.5)
    else:
        # bucket rows by group key and plot one line per bucket
        groups = {}
        for row in rows:
            groups.setdefault(row[group], []).append(row)
        for label, items in groups.items():
            xs = np.asarray([row[x] for row in items], dtype=np.float64)
            ys = np.asarray([row[y] for row in items], dtype=np.float64)
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker=marker, lw=1.5, label=str(label))
        # legend only when readable
        if len(groups) <= 10:
            ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)
    return ax.figure, ax

# thin wrapper that picks a sensible default title for the canonical beta sweep
def plot_difficulty_curve(rows, beta_key="algorithm_beta", metric="success_rate", group=None, ax=None, title=None):
    title = title or f"{metric} vs {beta_key}"
    return plot_grouped_metric(rows, x=beta_key, y=metric, group=group, ax=ax, title=title)

# create a default ax if the caller didn't pass one
def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return ax
    return ax
