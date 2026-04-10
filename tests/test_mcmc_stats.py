# ACF / autocov / ESS / tau_int / R-hat on canonical inputs (iid normal + AR(1))
import numpy as np

from spinglass.diagnostics.mcmc_stats import acf, autocov, ess, integrated_autocorr_time, rhat

# autocov at lag 0 must equal the sample variance
def test_autocov_zero_lag_is_variance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    g = autocov(x)
    assert abs(g[0] - float(np.var(x))) < 1e-10

# normalized acf at lag 0 must be exactly 1.0
def test_acf_zero_lag_is_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    rho = acf(x)
    # 1e-12: literal division of var by itself
    assert abs(rho[0] - 1.0) < 1e-12

# iid samples should give tau_int close to 1
def test_tau_int_near_one_for_iid():
    rng = np.random.default_rng(2)
    chains = rng.standard_normal((6, 4000))
    tau = integrated_autocorr_time(chains)
    # 1.5 upper bound: leaves slack for finite-sample noise but flags any bias
    assert tau < 1.5, f"iid tau too large: {tau}"

# AR(1) with phi=0.8: tau_int should sit in (3, 2 * theoretical)
def test_tau_int_larger_for_ar1():
    rng = np.random.default_rng(3)
    phi = 0.8
    n_chains, n_draws = 6, 4000
    chains = np.zeros((n_chains, n_draws))
    for c in range(n_chains):
        x = 0.0
        for t in range(n_draws):
            # stationary variance preserved by sqrt(1 - phi^2) noise scale
            x = phi * x + np.sqrt(1 - phi * phi) * rng.standard_normal()
            chains[c, t] = x
    tau = integrated_autocorr_time(chains)
    theoretical = (1 + phi) / (1 - phi)
    # window: lower bound 3 rules out iid; upper 2x theoretical absorbs estimator bias
    assert 3.0 < tau < 2 * theoretical, f"ar1 tau out of range: {tau} (theoretical ~{theoretical})"

# iid: ESS should be within 60% of total sample count and not exceed it
def test_ess_near_total_for_iid():
    rng = np.random.default_rng(4)
    chains = rng.standard_normal((4, 2000))
    total = 4 * 2000
    e = ess(chains)
    # 0.6 floor is loose enough for the truncation in integrated_autocorr_time
    assert 0.6 * total < e <= total, f"iid ess suspiciously off: {e}"

# strongly correlated AR(1) phi=0.9: ESS must be << total
def test_ess_much_smaller_for_ar1():
    rng = np.random.default_rng(5)
    phi = 0.9
    n_chains, n_draws = 6, 4000
    chains = np.zeros((n_chains, n_draws))
    for c in range(n_chains):
        x = 0.0
        for t in range(n_draws):
            x = phi * x + np.sqrt(1 - phi * phi) * rng.standard_normal()
            chains[c, t] = x
    e = ess(chains)
    total = n_chains * n_draws
    # 0.3 ceiling: theoretical ratio ~ (1-phi)/(1+phi) ~ 0.05
    assert e < 0.3 * total, f"ar1 ess too large: {e}"

# iid chains drawn from the same distribution should give rhat near 1
def test_rhat_near_one_for_iid():
    rng = np.random.default_rng(6)
    chains = rng.standard_normal((6, 1000))
    r = rhat(chains)
    # (0.95, 1.1) is the standard convergence window
    assert 0.95 < r < 1.1, f"iid rhat off: {r}"

# chains with very different means should produce rhat well above 1
def test_rhat_large_when_chains_differ():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000) + 5.0
    c = rng.standard_normal(1000) - 5.0
    d = rng.standard_normal(1000) + 2.5
    chains = np.stack([a, b, c, d], axis=0)
    r = rhat(chains)
    # 1.5 is well past the usual 1.01 / 1.1 thresholds — divergence is obvious
    assert r > 1.5, f"rhat should flag divergent chains: {r}"

# degenerate inputs (zero within-chain variance, too few draws) must return NaN
def test_rhat_nan_on_degenerate_input():
    assert np.isnan(rhat(np.ones((1, 500))))
    assert np.isnan(rhat(np.ones((2, 3))))

if __name__ == "__main__":
    test_autocov_zero_lag_is_variance()
    test_acf_zero_lag_is_one()
    test_tau_int_near_one_for_iid()
    test_tau_int_larger_for_ar1()
    test_ess_near_total_for_iid()
    test_ess_much_smaller_for_ar1()
    test_rhat_near_one_for_iid()
    test_rhat_large_when_chains_differ()
    test_rhat_nan_on_degenerate_input()
    print("test_mcmc_stats OK")
