# compute budget abstraction — convert sweeps / hamiltonian_evals to n_steps
# so discrete single-flip and relaxed all-spin updates compete on equal work
from dataclasses import dataclass

# steps: literal mcmc iterations
# sweeps: n flips (discrete) or n vector updates (relaxed)
# hamiltonian_evals: full energy/grad evaluations — most apples-to-apples
VALID_KINDS = ("steps", "sweeps", "hamiltonian_evals")

# immutable so a single Budget can be safely reused across runs
@dataclass(frozen=True)
class Budget:
    kind: str
    value: float

    # validate on construction; value>0 because zero-step runs are meaningless
    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Budget.kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if self.value <= 0:
            raise ValueError("Budget.value must be positive")

    # resolve to integer n_steps for the given problem size and space
    # cost_per_step: relaxed methods often touch all n spins per step,
    # so 1 relaxed step = n discrete flips at hamiltonian-eval cost
    def to_n_steps(self, n_spins, space, cost_per_step=1):
        if space not in ("discrete", "relaxed"):
            raise ValueError(f"space must be discrete or relaxed, got {space!r}")
        # literal steps — pass through
        if self.kind == "steps":
            return int(self.value)
        # one sweep = n single-spin flips in discrete, one vector update in relaxed
        if self.kind == "sweeps":
            if space == "discrete":
                return max(1, int(round(self.value * int(n_spins))))
            return max(1, int(round(self.value)))
        # hamiltonian_evals normalized by cost_per_step — discrete flip = 1/n eval
        if self.kind == "hamiltonian_evals":
            if space == "discrete":
                return max(1, int(round(self.value * int(n_spins) / max(cost_per_step, 1))))
            return max(1, int(round(self.value / max(cost_per_step, 1))))
        raise AssertionError("unreachable")

# convenience wrapper that also accepts raw ints/floats as step counts
def budget_to_n_steps(budget, n_spins, space):
    if budget is None:
        raise ValueError("budget cannot be None")
    if isinstance(budget, Budget):
        return budget.to_n_steps(n_spins, space)
    # raw int/float treated as literal step count
    if isinstance(budget, int):
        return budget
    if isinstance(budget, float):
        return int(budget)
    raise TypeError(f"budget must be Budget or int, got {type(budget).__name__}")

# factory: budget measured in sweeps
def sweeps(n):
    return Budget(kind="sweeps", value=float(n))

# factory: budget measured in raw mcmc steps
def steps(n):
    return Budget(kind="steps", value=float(n))

# factory: budget measured in full hamiltonian evaluations (most fair)
def hamiltonian_evals(n):
    return Budget(kind="hamiltonian_evals", value=float(n))
