"""sanity tests for diagnostics on synthetic chains with known structure.

uses data where the true ESS / R-hat / tau_int are either known analytically
or easy to bound, so regressions in the diagnostics layer get caught fast."""
import numpy as np

from spinglass.diagnostics.mcmc_stats import acf, autocov, ess, integrated_autocorr_time, rhat


def test_autocov_zero_lag_is_variance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    g = autocov(x)
    # autocov uses 1/n normalization, so gamma[0] = var(x) (biased)
    assert abs(g[0] - float(np.var(x))) < 1e-10


def test_acf_zero_lag_is_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    rho = acf(x)
    assert abs(rho[0] - 1.0) < 1e-12


def test_tau_int_near_one_for_iid():
    rng = np.random.default_rng(2)
    chains = rng.standard_normal((6, 4000))
    tau = integrated_autocorr_time(chains)
    # iid data should give tau ~ 1
    assert tau < 1.5, f"iid tau too large: {tau}"


def test_tau_int_larger_for_ar1():
    # AR(1): x_t = phi x_{t-1} + noise; theoretical tau_int = (1+phi)/(1-phi)
    rng = np.random.default_rng(3)
    phi = 0.8
    n_chains, n_draws = 6, 4000
    chains = np.zeros((n_chains, n_draws))
    for c in range(n_chains):
        x = 0.0
        for t in range(n_draws):
            x = phi * x + np.sqrt(1 - phi * phi) * rng.standard_normal()
            chains[c, t] = x
    tau = integrated_autocorr_time(chains)
    theoretical = (1 + phi) / (1 - phi)  # ~9
    # allow loose bound since truncated window estimator underestimates a bit
    assert 3.0 < tau < 2 * theoretical, f"ar1 tau out of range: {tau} (theoretical ~{theoretical})"


def test_ess_near_total_for_iid():
    rng = np.random.default_rng(4)
    chains = rng.standard_normal((4, 2000))
    total = 4 * 2000
    e = ess(chains)
    assert 0.6 * total < e <= total, f"iid ess suspiciously off: {e}"


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
    assert e < 0.3 * total, f"ar1 ess too large: {e}"


def test_rhat_near_one_for_iid():
    rng = np.random.default_rng(6)
    chains = rng.standard_normal((6, 1000))
    r = rhat(chains)
    assert 0.95 < r < 1.1, f"iid rhat off: {r}"


def test_rhat_large_when_chains_differ():
    # chains drawn from different means should produce rhat >> 1
    rng = np.random.default_rng(7)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000) + 5.0
    c = rng.standard_normal(1000) - 5.0
    d = rng.standard_normal(1000) + 2.5
    chains = np.stack([a, b, c, d], axis=0)
    r = rhat(chains)
    assert r > 1.5, f"rhat should flag divergent chains: {r}"


def test_rhat_nan_on_degenerate_input():
    # fewer than 2 chains or fewer than 4 draws -> NaN
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
