"""computational-budget conventions for discrete vs relaxed comparisons.

the project proposal calls out that matched compute is central to the
discrete-vs-relaxed story — otherwise the difficulty curves are incomparable.
this module fixes a single convention: a Budget object names the budget kind
(sweeps / steps / hamiltonian evaluations) and a value, and resolves to an
integer n_steps given the model size and algorithm space.

rationale: one discrete single-spin Metropolis step touches 1 spin, while one
Langevin / Adam / HMC step touches all n spins via a gradient evaluation. if
you match raw step counts, relaxed methods effectively get n times more
compute per step. "sweeps" rebalances this by defining 1 sweep = n single-
spin updates for discrete methods, 1 full gradient step for relaxed methods."""
from dataclasses import dataclass


VALID_KINDS = ("steps", "sweeps", "hamiltonian_evals")


@dataclass(frozen=True)
class Budget:
    """compute budget for a single run.

    kind:
      "steps"            raw n_steps passed straight through (least fair,
                         but useful for debugging a single algorithm)
      "sweeps"           one sweep = n_spins single-flip updates (discrete)
                         or one full gradient step (relaxed). matched compute
                         unit recommended for discrete-vs-relaxed plots.
      "hamiltonian_evals" total number of H(s) / energy_and_grad(x) calls.
                         closest to raw CPU cost for methods that dominate
                         on energy evaluation.
    value: numeric budget in the chosen unit."""
    kind: str
    value: float

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Budget.kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if self.value <= 0:
            raise ValueError("Budget.value must be positive")

    def to_n_steps(self, n_spins, space, cost_per_step=1):
        """resolve this budget to an integer n_steps for the given model/space.

        n_spins      number of spins in the model (used for sweeps / evals)
        space        "discrete" or "relaxed"
        cost_per_step  optional override for hamiltonian_evals mode"""
        if space not in ("discrete", "relaxed"):
            raise ValueError(f"space must be discrete or relaxed, got {space!r}")
        if self.kind == "steps":
            return int(self.value)
        if self.kind == "sweeps":
            if space == "discrete":
                return max(1, int(round(self.value * int(n_spins))))
            # one gradient step touches all n spins already, so one sweep = one step
            return max(1, int(round(self.value)))
        if self.kind == "hamiltonian_evals":
            if space == "discrete":
                # single-spin Metropolis ~1/n of a full H eval via O(1) incremental updates
                return max(1, int(round(self.value * int(n_spins) / max(cost_per_step, 1))))
            return max(1, int(round(self.value / max(cost_per_step, 1))))
        raise AssertionError("unreachable")


def budget_to_n_steps(budget, n_spins, space):
    """convenience: accept either a Budget or a plain int n_steps.

    plain ints pass through unchanged so legacy callers keep working."""
    if budget is None:
        raise ValueError("budget cannot be None")
    if isinstance(budget, Budget):
        return budget.to_n_steps(n_spins, space)
    if isinstance(budget, int):
        return budget
    if isinstance(budget, float):
        return int(budget)
    raise TypeError(f"budget must be Budget or int, got {type(budget).__name__}")


def sweeps(n):
    """shorthand constructor: Budget(kind='sweeps', value=n)."""
    return Budget(kind="sweeps", value=float(n))


def steps(n):
    """shorthand constructor: Budget(kind='steps', value=n)."""
    return Budget(kind="steps", value=float(n))


def hamiltonian_evals(n):
    """shorthand constructor: Budget(kind='hamiltonian_evals', value=n)."""
    return Budget(kind="hamiltonian_evals", value=float(n))
