# mcmc mixing diagnostics: autocov, acf, tau_int, ess, gelman-rubin r-hat
import numpy as np

# biased sample autocovariance gamma(lag) = (1/n) sum_{t} (x_t - mean)(x_{t+lag} - mean)
def autocov(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = x.size
    xc = x - np.mean(x)
    out = np.empty(n, dtype=np.float64)
    # naive O(n^2) loop; fine for the chain lengths we use here
    for lag in range(n):
        out[lag] = np.dot(xc[: n - lag], xc[lag:]) / n
    return out

# normalized autocorrelation rho(lag) = gamma(lag) / gamma(0)
def acf(x, max_lag=None):
    gamma = autocov(x)
    # degenerate chain (constant) -> treat as fully correlated
    if gamma[0] <= 0:
        limit = gamma.size if max_lag is None else max_lag + 1
        return np.ones(limit, dtype=np.float64)
    rho = gamma / gamma[0]
    if max_lag is None:
        return rho
    return rho[: max_lag + 1]

# integrated autocorrelation time via geyer's initial-positive sequence
# tau_int = 1 + 2 sum_{k>=1} rho(k), truncated when consecutive pairs go nonpositive
def integrated_autocorr_time(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    # not enough draws to estimate -> conservative tau = 1
    if t < 3:
        return 1.0
    # average autocov across chains then normalize once at the end
    acovs = np.zeros((c, t), dtype=np.float64)
    for i in range(c):
        acovs[i] = autocov(x[i])
    mean_acov = np.mean(acovs, axis=0)
    if mean_acov[0] <= 0:
        return 1.0
    rho = mean_acov / mean_acov[0]
    # sum positive pairs (rho_{2k-1} + rho_{2k}); break when pair turns nonpositive
    pair_sum = 0.0
    k = 1
    while 2 * k < t:
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair <= 0:
            break
        pair_sum += pair
        k += 1
    # floor at 1.0; tau_int < 1 is meaningless for positively autocorrelated chains
    return float(max(1.0, 1.0 + 2.0 * pair_sum))

# effective sample size: N_total / tau_int  (each tau_int draws ~ one independent draw)
def ess(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    tau = integrated_autocorr_time(x)
    # clamp to [1, c*t]: cannot exceed raw count, cannot drop below one effective draw
    return float(max(1.0, min(c * t, (c * t) / tau)))

# gelman-rubin r-hat with split chains; ratio of pooled-to-within variance
def rhat(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("chains_2d must have shape (n_chains, n_draws)")
    c, t = x.shape
    # need >=2 chains and enough draws to split each in half
    if c < 2 or t < 4:
        return np.nan
    half = t // 2
    if half < 2:
        return np.nan
    # drop trailing odd draw so the two halves match
    x = x[:, : 2 * half]
    # split each chain at the midpoint -> 2c chains of length n=half
    split = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    m, n = split.shape
    means = np.mean(split, axis=1)
    vars_ = np.var(split, axis=1, ddof=1)
    # within-chain variance W and between-chain variance B
    W = np.mean(vars_)
    if W <= 0:
        return np.nan
    B = n * np.var(means, ddof=1)
    # estimator of marginal posterior variance combining W and B
    var_hat = ((n - 1) / n) * W + B / n
    # r-hat -> 1 as chains agree; values >> 1 flag non-convergence
    return float(np.sqrt(var_hat / W))
