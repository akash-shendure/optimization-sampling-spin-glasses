"""diagnostic plots for traces and multi-chain summaries."""
import numpy as np
import matplotlib.pyplot as plt

from ..diagnostics.mcmc_stats import acf


def plot_trace(chains_2d, x=None, ax=None, title=None, ylabel="value", alpha=0.9):
    chains = np.asarray(chains_2d, dtype=np.float64)
    if chains.ndim == 1:
        chains = chains[None, :]
    ax = _get_ax(ax)
    xx = np.arange(chains.shape[1]) if x is None else np.asarray(x)
    for i, chain in enumerate(chains):
        ax.plot(xx, chain, lw=1.0, alpha=alpha, label=f"chain {i}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if chains.shape[0] <= 8:
        ax.legend()
    return ax.figure, ax


def plot_acf(chains_2d, max_lag=100, ax=None, title=None, alpha=0.9):
    chains = np.asarray(chains_2d, dtype=np.float64)
    if chains.ndim == 1:
        chains = chains[None, :]
    ax = _get_ax(ax)
    lags = np.arange(max_lag + 1)
    for i, chain in enumerate(chains):
        ax.plot(lags, acf(chain, max_lag=max_lag), lw=1.0, alpha=alpha, label=f"chain {i}")
    ax.set_xlabel("lag")
    ax.set_ylabel("acf")
    ax.set_ylim(-0.2, 1.0)
    if title is not None:
        ax.set_title(title)
    if chains.shape[0] <= 8:
        ax.legend()
    return ax.figure, ax


def plot_rank_histogram(chains_2d, bins=20, ax=None, title=None, density=True):
    chains = np.asarray(chains_2d, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    ranks = np.argsort(np.argsort(chains.reshape(-1))).reshape(chains.shape)
    ax = _get_ax(ax)
    for i, chain in enumerate(ranks):
        ax.hist(chain, bins=bins, alpha=0.45, density=density, label=f"chain {i}")
    ax.set_xlabel("rank")
    ax.set_ylabel("density" if density else "count")
    if title is not None:
        ax.set_title(title)
    if chains.shape[0] <= 8:
        ax.legend()
    return ax.figure, ax


def plot_pair(x, y, ax=None, title=None, xlabel="x", ylabel="y", alpha=0.35, s=8):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    ax = _get_ax(ax)
    for i in range(x.shape[0]):
        ax.scatter(x[i], y[i], alpha=alpha, s=s, label=f"chain {i}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if x.shape[0] <= 8:
        ax.legend()
    return ax.figure, ax


def plot_pair_matrix(data_dict, names=None, title=None, n_per_series=800, alpha=0.3, s=6):
    if not data_dict:
        raise ValueError("data_dict must be non-empty")
    if names is None:
        names = list(data_dict.keys())
    arrays = [np.asarray(data_dict[name], dtype=np.float64) for name in names]
    arrays = [_to_2d(arr) for arr in arrays]
    n_chains, n_draws = arrays[0].shape
    idx = np.linspace(0, n_draws - 1, min(n_per_series, n_draws)).astype(int)

    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(3.2 * k, 3.2 * k), squeeze=False)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                pooled = arrays[i][:, idx].reshape(-1)
                ax.hist(pooled, bins=25, density=True, alpha=0.8)
            else:
                for c in range(n_chains):
                    ax.scatter(arrays[j][c, idx], arrays[i][c, idx], s=s, alpha=alpha)
            if i == k - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return ax
    return ax


def _to_2d(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError("arrays must be 1d or 2d")
    return x
