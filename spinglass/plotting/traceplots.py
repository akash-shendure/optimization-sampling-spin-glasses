"""high-level trace plots for optimizer and sampler outputs."""
import numpy as np
import matplotlib.pyplot as plt


def plot_optimizer_trace(record, x="step", y="energy", best=True, ax=None, title=None):
    trace = record["trace"]
    if trace is None:
        raise ValueError("record has no trace")
    ax = _get_ax(ax)
    xx = np.asarray(trace[x], dtype=np.float64)
    yy = np.asarray(trace[y], dtype=np.float64)
    ax.plot(xx, yy, label=y, lw=1.5)
    if best and "best_energy" in trace and y == "energy":
        ax.plot(xx, np.asarray(trace["best_energy"], dtype=np.float64), label="best_energy", lw=1.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{record['meta']['algorithm_name']} {y}")
    if best and "best_energy" in trace and y == "energy":
        ax.legend()
    return ax.figure, ax


def plot_sampler_trace(records, observable="energy", x="step", ax=None, title=None):
    if not records:
        raise ValueError("records must be non-empty")
    ax = _get_ax(ax)
    for record in records:
        trace = record["trace"]
        if trace is None or observable not in trace:
            continue
        xx = np.asarray(trace[x], dtype=np.float64)
        yy = np.asarray(trace[observable], dtype=np.float64)
        chain_id = record["meta"].get("chain_id")
        label = f"chain {chain_id}" if chain_id is not None else record["meta"]["algorithm_name"]
        ax.plot(xx, yy, lw=1.2, alpha=0.85, label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(observable)
    ax.set_title(title or f"{records[0]['meta']['algorithm_name']} {observable}")
    labels = [line.get_label() for line in ax.get_lines()]
    if len(labels) <= 8:
        ax.legend()
    return ax.figure, ax


def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        return ax
    return ax

