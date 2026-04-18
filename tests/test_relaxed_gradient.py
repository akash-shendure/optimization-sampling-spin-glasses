"""RelaxedHamiltonian analytic gradient vs central finite differences.

if grad is wrong, all relaxed optimizers/samplers (GD, Adam, Langevin, MALA,
HMC) will diverge from the intended target, so this is a high-value invariant
to lock down before trusting any sweeps."""
import numpy as np

from spinglass import EdwardsAnderson2D, RelaxedHamiltonian, SherringtonKirkpatrick


def _finite_diff_grad(H, x, eps=1e-5):
    g = np.zeros_like(x)
    for k in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        g[k] = (H.energy(xp) - H.energy(xm)) / (2.0 * eps)
    return g


def _check_grad(H, x, rtol=1e-5, atol=1e-6):
    g_analytic = H.grad(x)
    g_fd = _finite_diff_grad(H, x)
    assert np.allclose(g_analytic, g_fd, rtol=rtol, atol=atol), (
        f"analytic grad vs fd grad off: max abs diff {np.max(np.abs(g_analytic - g_fd))}"
    )


def test_grad_on_ea_alpha_1_lambda_0():
    model = EdwardsAnderson2D(L=4, disorder="pm1")
    H = RelaxedHamiltonian(model, alpha=1.0, lam=0.0)
    rng = np.random.default_rng()
    for _ in range(3):
        x = 0.5 * rng.standard_normal(model.n)
        _check_grad(H, x)


def test_grad_on_ea_with_lambda_penalty():
    model = EdwardsAnderson2D(L=4, disorder="gaussian")
    H = RelaxedHamiltonian(model, alpha=1.5, lam=0.25)
    rng = np.random.default_rng()
    for _ in range(3):
        x = 0.3 * rng.standard_normal(model.n)
        _check_grad(H, x, rtol=1e-4, atol=1e-5)


def test_grad_on_sk_dense():
    model = SherringtonKirkpatrick(n=12)
    H = RelaxedHamiltonian(model, alpha=1.0, lam=0.1)
    rng = np.random.default_rng()
    x = 0.4 * rng.standard_normal(model.n)
    _check_grad(H, x)


def test_energy_and_grad_matches_separate_calls():
    model = EdwardsAnderson2D(L=5, disorder="pm1")
    H = RelaxedHamiltonian(model, alpha=1.2, lam=0.05)
    rng = np.random.default_rng()
    x = 0.6 * rng.standard_normal(model.n)
    E1 = H.energy(x)
    g1 = H.grad(x)
    E2, g2 = H.energy_and_grad(x)
    assert abs(E1 - E2) < 1e-12
    assert np.allclose(g1, g2, atol=1e-12)


def test_project_gives_pm_one_int8():
    rng = np.random.default_rng()
    x = rng.standard_normal(32)
    s = RelaxedHamiltonian.project(x)
    assert s.dtype == np.int8
    assert set(np.unique(s).tolist()) <= {-1, 1}
    # zero x should tiebreak to +1
    assert RelaxedHamiltonian.project(np.array([0.0, 0.0]))[0] == 1


def test_energy_matches_discrete_at_projected_extremes():
    # as alpha -> large and x is binary-ish, relaxed energy (lam=0) collapses
    # onto discrete energy of the sign pattern
    from spinglass import DiscreteHamiltonian

    model = EdwardsAnderson2D(L=5)
    Hd = DiscreteHamiltonian(model)
    Hr = RelaxedHamiltonian(model, alpha=8.0, lam=0.0)
    rng = np.random.default_rng()
    x = 2.0 * (rng.integers(0, 2, size=model.n).astype(np.float64) - 0.5)
    E_rel = Hr.energy(x)
    s = RelaxedHamiltonian.project(x)
    E_dis = Hd.energy(s)
    assert abs(E_rel - E_dis) < 1e-4


if __name__ == "__main__":
    test_grad_on_ea_alpha_1_lambda_0()
    test_grad_on_ea_with_lambda_penalty()
    test_grad_on_sk_dense()
    test_energy_and_grad_matches_separate_calls()
    test_project_gives_pm_one_int8()
    test_energy_matches_discrete_at_projected_extremes()
    print("test_relaxed_gradient OK")
