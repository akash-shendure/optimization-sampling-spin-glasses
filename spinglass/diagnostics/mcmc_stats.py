"""basic multi-chain diagnostics."""
import numpy as np


def autocov(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = x.size
    xc = x - np.mean(x)
    out = np.empty(n, dtype=np.float64)
    for lag in range(n):
        out[lag] = np.dot(xc[: n - lag], xc[lag:]) / n
    return out


def acf(x, max_lag=None):
    gamma = autocov(x)
    if gamma[0] <= 0:
        limit = gamma.size if max_lag is None else max_lag + 1
        return np.ones(limit, dtype=np.float64)
    rho = gamma / gamma[0]
    if max_lag is None:
        return rho
    return rho[: max_lag + 1]


def integrated_autocorr_time(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    if t < 3:
        return 1.0
    acovs = np.zeros((c, t), dtype=np.float64)
    for i in range(c):
        acovs[i] = autocov(x[i])
    mean_acov = np.mean(acovs, axis=0)
    if mean_acov[0] <= 0:
        return 1.0
    rho = mean_acov / mean_acov[0]
    pair_sum = 0.0
    k = 1
    while 2 * k < t:
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair <= 0:
            break
        pair_sum += pair
        k += 1
    return float(max(1.0, 1.0 + 2.0 * pair_sum))


def ess(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    tau = integrated_autocorr_time(x)
    return float(max(1.0, min(c * t, (c * t) / tau)))


def rhat(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    if c < 2 or t < 4:
        return np.nan
    half = t // 2
    if half < 2:
        return np.nan
    x = x[:, : 2 * half]
    split = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    m, n = split.shape
    means = np.mean(split, axis=1)
    vars_ = np.var(split, axis=1, ddof=1)
    W = np.mean(vars_)
    if W <= 0:
        return np.nan
    B = n * np.var(means, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))

