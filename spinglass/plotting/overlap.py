"""plots for replica-overlap diagnostics in glassy models.

pairs with experiments/overlap.py. the overlap distribution P(q) is the main
empirical fingerprint of glassy vs paramagnetic vs ordered phases, so these
plots are the dual of the temperature difficulty curve."""
import numpy as np
import matplotlib.pyplot as plt


def plot_overlap_histogram(overlap_rows, ax=None, title=None, color="#1f77b4"):
    """plot P(q) for a single condition from experiments.overlap_histogram."""
    if not overlap_rows:
        raise ValueError("overlap_rows must be non-empty")
    row = overlap_rows[0] if isinstance(overlap_rows, list) else overlap_rows
    edges = np.asarray(row["bin_edges"], dtype=np.float64)
    density = np.asarray(row["density"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax = _get_ax(ax)
    ax.bar(centers, density, width=width, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("overlap q")
    ax.set_ylabel("P(q)")
    ax.set_xlim(-1.05, 1.05)
    if title is not None:
        ax.set_title(title)
    return ax.figure, ax


def plot_overlap_histograms_by_beta(overlap_rows, beta_key="algorithm_beta", ncols=3, title=None):
    """small-multiples grid: one P(q) panel per temperature."""
    rows = sorted(overlap_rows, key=lambda r: float(r[beta_key]))
    n = len(rows)
    if n == 0:
        raise ValueError("no overlap rows to plot")
    ncols = min(int(ncols), n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        plot_overlap_histogram([row], ax=ax, title=f"{beta_key}={row[beta_key]}")
    for ax in axes[n:]:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def plot_mean_abs_q_curve(overlap_rows, beta_key="algorithm_beta", ax=None, title=None, marker="o"):
    """mean |q| vs beta — a scalar summary of the overlap distribution.

    paramagnetic phase: near 0. ordered / glassy phase: grows toward 1.
    the knee in this curve is the easiest scalar fingerprint of a phase
    transition visible to a sampler."""
    if not overlap_rows:
        raise ValueError("overlap_rows must be non-empty")
    xs = np.asarray([float(row[beta_key]) for row in overlap_rows], dtype=np.float64)
    ys = np.asarray([float(row["mean_abs_q"]) for row in overlap_rows], dtype=np.float64)
    order = np.argsort(xs)
    ax = _get_ax(ax)
    ax.plot(xs[order], ys[order], marker=marker, lw=1.6)
    ax.set_xlabel(beta_key)
    ax.set_ylabel("<|q|>")
    ax.set_ylim(-0.05, 1.05)
    if title is not None:
        ax.set_title(title)
    return ax.figure, ax


def plot_overlap_vs_energy(overlap_rows, energy_rows, beta_key="algorithm_beta", ax=None, title=None):
    """scatter <|q|> against mean energy per condition.

    matches rows by beta_key; the resulting curve shows the joint
    behavior of the two main scalar diagnostics across the sweep."""
    e_map = {float(row[beta_key]): float(row.get("mean_energy", np.nan)) for row in energy_rows}
    xs, ys, labels = [], [], []
    for row in overlap_rows:
        k = float(row[beta_key])
        if k in e_map and np.isfinite(e_map[k]):
            xs.append(e_map[k])
            ys.append(float(row["mean_abs_q"]))
            labels.append(k)
    if not xs:
        raise ValueError("no matching energy/overlap rows")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    order = np.argsort(xs)
    ax = _get_ax(ax)
    ax.plot(xs[order], ys[order], marker="o", lw=1.6)
    for x, y, k in zip(xs, ys, labels):
        ax.annotate(f"{k:g}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)
    ax.set_xlabel("mean energy")
    ax.set_ylabel("<|q|>")
    if title is not None:
        ax.set_title(title)
    return ax.figure, ax


def _get_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.6))
    return ax
