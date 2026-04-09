# Optimization and Sampling in Spin Glasses

Research code for a Statistics 221 project on optimization and sampling in rugged spin-system energy landscapes. The central question is whether algorithmic performance exhibits a computational phase transition as temperature, disorder, and interaction topology change.

We compare:
- discrete methods on `s in {-1, +1}^n`
- relaxed continuous methods on `x in R^n` using `tanh(alpha x)` as a surrogate spin representation

The repository currently includes:
- model and coupling builders
- discrete and relaxed Hamiltonians
- optimization algorithms
- sampling algorithms
- experiment runners
- benchmark summaries
- plotting utilities

## Overview

We study four model families with increasing structural complexity:
- `IsingFerromagnet2D`: clean lattice baseline with classical ordering
- `EdwardsAnderson2D`: frustrated 2D lattice glass
- `SparseRandomGlass`: sparse Erdos-Renyi interaction graph
- `SherringtonKirkpatrick`: dense mean-field SK model

The main empirical goal is to compare how optimization and sampling difficulty changes with:
- inverse temperature `beta`
- system size
- disorder realization
- interaction topology

For optimization, the key outcomes are things like:
- best energy found under fixed budget
- projected discrete energy for relaxed methods
- success rate or time-to-threshold

For sampling, the key outcomes are things like:
- mean energy
- magnetization in ferromagnetic models
- overlap-based observables in glassy models
- ESS, `R-hat`, autocorrelation, acceptance, and swap statistics

The package is designed so algorithms do not just return a final answer; they also return traces and metadata suitable for benchmarking and diagnostics.

## Repository Layout

```text
spinglass/
  models/        model containers for topology + disorder realization
  couplings/     low-level builders for coupling matrices J
  hamiltonian/   energy, local field, gradient, and projection logic
  optimizers/    discrete and relaxed optimization algorithms
  samplers/      discrete and relaxed sampling algorithms
  diagnostics/   ESS, R-hat, ACF, overlap, and summary helpers
  experiments/   single-run and sweep runners, grouped benchmark summaries
  plotting/      matplotlib helpers for traces, diagnostics, difficulty curves
  utils/         RNG helpers and shared trace / spin update utilities
```

The main design convention is:
- a `model` stores topology, disorder, and metadata
- a `Hamiltonian` performs the physics/math on top of that model
- algorithms operate on a Hamiltonian
- experiments run algorithms repeatedly and collect analysis-ready summaries

## Core Conventions and Assumptions

- The discrete Hamiltonian is
  `H(s) = -0.5 * s^T J s`
  with symmetric zero-diagonal `J`.
- Discrete states are `int8` arrays in `{-1, +1}`.
- Relaxed methods use `t = tanh(alpha x)` and the surrogate energy implemented in `RelaxedHamiltonian`.
- `J` may be sparse CSR or dense `ndarray`; downstream code should prefer `J @ v`.
- Models are lightweight and stateless apart from stored RNG/disorder information.
- Hamiltonians are also stateless wrappers; samplers are responsible for maintaining local fields incrementally when needed.
- Relaxed methods can optionally project back to discrete space via `RelaxedHamiltonian.project(x)`.

Important implementation detail:
- discrete single-spin methods rely on `delta_energy` and incremental updates of `h = J @ s` for efficiency

## Dependencies

There is not yet a formal `requirements.txt` or `pyproject.toml`. The current code uses:
- `numpy`
- `scipy`
- `matplotlib` for plotting helpers

You should make sure those packages are available in your environment before running experiments.

## Usage

### Build a model and Hamiltonians

```python
from spinglass import (
    EdwardsAnderson2D,
    DiscreteHamiltonian,
    RelaxedHamiltonian,
)

model = EdwardsAnderson2D(L=16, disorder="pm1", seed=0)
Hd = DiscreteHamiltonian(model)
Hr = RelaxedHamiltonian(model, alpha=1.0, lam=0.1)
```

### Run an optimizer

```python
import numpy as np
from spinglass import SimulatedAnnealing

opt = SimulatedAnnealing(
    Hd,
    beta_schedule=np.linspace(0.1, 2.0, 5000),
    seed=0,
)
result = opt.run(n_steps=5000, trace_every=50)

print(result["summary"]["best_energy"])
```

### Run a sampler

```python
from spinglass import MetropolisSampler

sampler = MetropolisSampler(Hd, beta=0.8, seed=0)
result = sampler.run(
    n_steps=10000,
    burn_in=2000,
    thin=10,
    trace_every=100,
    store_samples=True,
)

print(result["summary"]["acceptance_rate"])
```

### Run a sweep with the experiment layer

```python
import numpy as np
from spinglass import (
    EdwardsAnderson2D,
    SimulatedAnnealing,
    run_grid,
    summarize_optimization_table,
)

results = run_grid(
    task="optimization",
    space="discrete",
    model_class=EdwardsAnderson2D,
    model_grid={"L": [8], "seed": [0, 1, 2]},
    algorithm_class=SimulatedAnnealing,
    algorithm_grid={"beta_schedule": [np.linspace(0.1, 2.0, 2000)], "seed": [0]},
    run_kwargs={"n_steps": 2000, "trace_every": 50},
    n_restarts=3,
    keep_trace=False,
    experiment_name="ea_sa_demo",
)

table = results["table"]
summary_rows = summarize_optimization_table(table, group_by=["model_seed"])
```

### Make a diagnostic or difficulty plot

```python
from spinglass import plot_difficulty_curve

fig, ax = plot_difficulty_curve(
    summary_rows,
    beta_key="algorithm_beta",
    metric="mean_best_energy",
)
```

## Implemented Methods

### Optimizers

Discrete:
- `GreedySpinDescent`
- `SimulatedAnnealing`
- `ParallelTemperingOptimizer`

Relaxed:
- `GradientDescentOptimizer`
- `AdamOptimizer`

### Samplers

Discrete:
- `MetropolisSampler`
- `GibbsSampler`
- `ParallelTemperingSampler`

Relaxed:
- `LangevinSampler`
- `MALASampler`

Each algorithm exposes a lightweight constructor and a `run(...)` method.

## Return Format

All optimizer and sampler `run(...)` methods return a dict with the shared shape:

```python
{
    "summary": {...},
    "trace": {...},
    "artifacts": {...},
}
```

Typical contents:
- `summary`: scalar metrics like runtime, final energy, best energy, acceptance rate
- `trace`: downsampled curves like `step`, `time_sec`, `energy`, `best_energy`, `grad_norm`
- `artifacts`: heavier outputs like final state, best state, or stored samples

This common format is what the experiment and plotting layers build on.

## Experiment Workflow

The intended workflow is:

1. choose a model family and parameter grid
2. choose an algorithm and hyperparameter grid
3. run `run_single(...)` for one job or `run_grid(...)` for a sweep
4. inspect `results["table"]` for per-run summaries
5. use `summarize_optimization_table(...)` or `summarize_sampling_table(...)` for grouped benchmark summaries
6. use the plotting helpers for trace diagnostics and difficulty curves

Useful experiment functions:
- `run_single(...)`
- `run_grid(...)`
- `flatten_record(...)`
- `parameter_grid(...)`
- `summarize_optimization_table(...)`
- `summarize_sampling_table(...)`

## Diagnostics and Plotting

Implemented diagnostics include:
- autocovariance / ACF
- ESS
- split `R-hat`
- integrated autocorrelation time
- magnetization
- overlap and pairwise overlaps

Implemented plotting helpers include:
- optimizer trace plots
- sampler trace plots
- generic trace plots
- ACF plots
- rank histograms
- pair plots and pair matrices
- grouped metric plots
- difficulty curves

## Notes

- The code is organized around a model / Hamiltonian / algorithm separation.
- Most public functionality is re-exported from `spinglass/__init__.py`.
- New methods should ideally preserve the current `summary` / `trace` / `artifacts` output pattern so they work naturally with the experiment and plotting layers.
