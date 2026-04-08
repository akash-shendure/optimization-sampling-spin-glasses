"""plots for grouped benchmark summaries."""
import numpy as np
import matplotlib.pyplot as plt


def plot_grouped_metric(rows, x, y, group=None, ax=None, title=None, marker="o"):
    if not rows:
        raise ValueError("rows must be non-empty")
    ax = _get_ax(ax)
    if group is None:
        xs = np.asarray([row[x] for row in rows], dtype=np.float64)
        ys = np.asarray([row[y] for row in rows], dtype=np.float64)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=marker, lw=1.5)
    else:
        groups = {}
        for row in rows:
            groups.setdefault(row[group], []).append(row)
        for label, items in groups.items():
            xs = np.asarray([row[x] for row in items], dtype=np.float64)
            ys = np.asarray([row[y] for row in items], dtype=np.float64)
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker=marker, lw=1.5, label=str(label))
        if len(groups) <= 10:
            ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)
    return ax.figure, ax


def plot_difficulty_curve(rows, beta_key="algorithm_beta", metric="success_rate", group=None, ax=None, title=None):
    title = title or f"{metric} vs {beta_key}"
    return plot_grouped_metric(rows, x=beta_key, y=metric, group=group, ax=ax, title=title)


def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return ax
    return ax

