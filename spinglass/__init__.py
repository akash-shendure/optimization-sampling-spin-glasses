"""spinglass: models, Hamiltonians, optimizers, samplers, and diagnostics."""
# top-level re-exports so user code can do `from spinglass import ...`
from .models.ising2d import IsingFerromagnet2D
from .models.edwards_anderson import EdwardsAnderson2D
from .models.sparse_glass import SparseRandomGlass
from .models.sk import SherringtonKirkpatrick
from .hamiltonian.discrete import DiscreteHamiltonian
from .hamiltonian.relaxed import RelaxedHamiltonian
from .optimizers.discrete_greedy import GreedySpinDescent
from .optimizers.simulated_annealing import SimulatedAnnealing
from .optimizers.parallel_tempering import ParallelTemperingOptimizer
from .optimizers.gradient_descent import GradientDescentOptimizer
from .optimizers.adam import AdamOptimizer
from .samplers.metropolis import MetropolisSampler
from .samplers.gibbs import GibbsSampler
from .samplers.parallel_tempering import ParallelTemperingSampler
from .samplers.langevin import LangevinSampler
from .samplers.mala import MALASampler
from .diagnostics.mcmc_stats import acf, autocov, ess, integrated_autocorr_time, rhat
from .diagnostics.observables import magnetization, overlap, pairwise_overlaps
from .diagnostics.summaries import summarize_optimizer_runs, summarize_sampler_runs

__all__ = [
    "IsingFerromagnet2D",
    "EdwardsAnderson2D",
    "SparseRandomGlass",
    "SherringtonKirkpatrick",
    "DiscreteHamiltonian",
    "RelaxedHamiltonian",
    "GreedySpinDescent",
    "SimulatedAnnealing",
    "ParallelTemperingOptimizer",
    "GradientDescentOptimizer",
    "AdamOptimizer",
    "MetropolisSampler",
    "GibbsSampler",
    "ParallelTemperingSampler",
    "LangevinSampler",
    "MALASampler",
    "acf",
    "autocov",
    "ess",
    "integrated_autocorr_time",
    "rhat",
    "magnetization",
    "overlap",
    "pairwise_overlaps",
    "summarize_optimizer_runs",
    "summarize_sampler_runs",
]
