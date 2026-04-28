# factories that turn class objects + kwargs into ready-to-run components
from ..hamiltonian.discrete import DiscreteHamiltonian
from ..hamiltonian.relaxed import RelaxedHamiltonian

# instantiate a model from class + kwargs — trivial but keeps run_single uniform
def build_model(model_class, **model_kwargs):
    return model_class(**model_kwargs)

# wrap a model in the appropriate hamiltonian for the chosen space
# alpha/lam/reg are only meaningful in the relaxed branch
def build_hamiltonian(model, space, alpha=2.0, lam=0.0, reg="linear"):
    if space == "discrete":
        return DiscreteHamiltonian(model)
    if space == "relaxed":
        return RelaxedHamiltonian(model, alpha=alpha, lam=lam, reg=reg)
    raise ValueError("space must be 'discrete' or 'relaxed'")

# attach an algorithm (optimizer or sampler) to a hamiltonian
def build_algorithm(algorithm_class, hamiltonian, **algorithm_kwargs):
    return algorithm_class(hamiltonian, **algorithm_kwargs)
