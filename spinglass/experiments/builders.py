"""construction helpers for experiment runners."""
from ..hamiltonian.discrete import DiscreteHamiltonian
from ..hamiltonian.relaxed import RelaxedHamiltonian


def build_model(model_class, **model_kwargs):
    return model_class(**model_kwargs)


def build_hamiltonian(model, space, alpha=1.0, lam=0.0):
    if space == "discrete":
        return DiscreteHamiltonian(model)
    if space == "relaxed":
        return RelaxedHamiltonian(model, alpha=alpha, lam=lam)
    raise ValueError("space must be 'discrete' or 'relaxed'")


def build_algorithm(algorithm_class, hamiltonian, **algorithm_kwargs):
    return algorithm_class(hamiltonian, **algorithm_kwargs)

