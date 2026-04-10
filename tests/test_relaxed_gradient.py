# gradient of RelaxedHamiltonian matches finite difference; covers linear + quartic reg
import numpy as np

from spinglass import EdwardsAnderson2D, RelaxedHamiltonian, SherringtonKirkpatrick

# central-difference gradient as ground truth; eps=1e-5 balances truncation vs roundoff
def _finite_diff_grad(H, x, eps=1e-5):
    g = np.zeros_like(x)
    for k in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        g[k] = (H.energy(xp) - H.energy(xm)) / (2.0 * eps)
    return g

# shared assertion: analytic grad must track FD grad within rtol/atol
def _check_grad(H, x, rtol=1e-5, atol=1e-6):
    g_analytic = H.grad(x)
    g_fd = _finite_diff_grad(H, x)
    assert np.allclose(g_analytic, g_fd, rtol=rtol, atol=atol), (
        f"analytic grad vs fd grad off: max abs diff {np.max(np.abs(g_analytic - g_fd))}"
    )

# pure interaction term, alpha=1, no penalty — exercises the J t branch only
def test_grad_on_ea_alpha_1_lambda_0():
    model = EdwardsAnderson2D(L=4, disorder="pm1", seed=0)
    H = RelaxedHamiltonian(model, alpha=1.0, lam=0.0)
    rng = np.random.default_rng(1)
    for _ in range(3):
        # 0.5 std keeps x in a regime where sech^2 isn't near zero
        x = 0.5 * rng.standard_normal(model.n)
        _check_grad(H, x)

# linear penalty active; loosen tolerances because lam term adds curvature
def test_grad_on_ea_with_lambda_penalty():
    model = EdwardsAnderson2D(L=4, disorder="gaussian", seed=2)
    H = RelaxedHamiltonian(model, alpha=1.5, lam=0.25)
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = 0.3 * rng.standard_normal(model.n)
        # rtol=1e-4, atol=1e-5 absorb the extra FD error from the penalty
        _check_grad(H, x, rtol=1e-4, atol=1e-5)

# dense SK to make sure the gradient path also handles non-sparse J
def test_grad_on_sk_dense():
    model = SherringtonKirkpatrick(n=12, seed=4)
    H = RelaxedHamiltonian(model, alpha=1.0, lam=0.1)
    rng = np.random.default_rng(5)
    x = 0.4 * rng.standard_normal(model.n)
    _check_grad(H, x)

# energy_and_grad must return exactly the same numbers as energy + grad called separately
def test_energy_and_grad_matches_separate_calls():
    model = EdwardsAnderson2D(L=5, disorder="pm1", seed=6)
    H = RelaxedHamiltonian(model, alpha=1.2, lam=0.05)
    rng = np.random.default_rng(7)
    x = 0.6 * rng.standard_normal(model.n)
    E1 = H.energy(x)
    g1 = H.grad(x)
    E2, g2 = H.energy_and_grad(x)
    # 1e-12: identical arithmetic, just one tanh evaluation instead of two
    assert abs(E1 - E2) < 1e-12
    assert np.allclose(g1, g2, atol=1e-12)

# project maps R^n -> {-1, +1}^n as int8; exact zeros must tiebreak to +1
def test_project_gives_pm_one_int8():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(32)
    s = RelaxedHamiltonian.project(x)
    assert s.dtype == np.int8
    assert set(np.unique(s).tolist()) <= {-1, 1}
    # zero-tiebreak: documented contract that sign(0) -> +1
    assert RelaxedHamiltonian.project(np.array([0.0, 0.0]))[0] == 1

# at very sharp alpha, relaxed energy on a +/-1 input should approach discrete energy
def test_energy_matches_discrete_at_projected_extremes():
    from spinglass import DiscreteHamiltonian

    model = EdwardsAnderson2D(L=5, seed=9)
    Hd = DiscreteHamiltonian(model)
    # alpha=8 saturates tanh hard so t ~ s, lam=0 kills the penalty term
    Hr = RelaxedHamiltonian(model, alpha=8.0, lam=0.0)
    rng = np.random.default_rng(10)
    x = 2.0 * (rng.integers(0, 2, size=model.n).astype(np.float64) - 0.5)
    E_rel = Hr.energy(x)
    s = RelaxedHamiltonian.project(x)
    E_dis = Hd.energy(s)
    # 1e-4: residual 1 - tanh(8)^2 ~ 2e-7 per term, sum scales with edges
    assert abs(E_rel - E_dis) < 1e-4

if __name__ == "__main__":
    test_grad_on_ea_alpha_1_lambda_0()
    test_grad_on_ea_with_lambda_penalty()
    test_grad_on_sk_dense()
    test_energy_and_grad_matches_separate_calls()
    test_project_gives_pm_one_int8()
    test_energy_matches_discrete_at_projected_extremes()
    print("test_relaxed_gradient OK")
