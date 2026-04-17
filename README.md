# Optimization and Sampling in Spin Glasses

Statistics 221 project on whether optimization and sampling algorithms exhibit a computational phase transition as spin-system landscapes move from clean to critical to glassy. We compare discrete methods on `s in {-1, +1}^n` against relaxed continuous methods on `x in R^n` using `tanh(alpha x)` as a surrogate, on four model families:

- `IsingFerromagnet2D` — clean lattice baseline with classical ordering
- `EdwardsAnderson2D` — frustrated 2D lattice glass
- `SparseRandomGlass` — sparse Erdos-Renyi interaction graph
- `SherringtonKirkpatrick` — dense mean-field SK

The central empirical object is a difficulty curve (best energy, sampling mixing error) against inverse temperature `beta`, swept across model families and system sizes.

## Install

```bash
pip install -e .
```

Requires `numpy`, `scipy`, `matplotlib`. Pytest is optional — the zero-dependency test runner works standalone.

## Quickstart

```python
from spinglass import EdwardsAnderson2D, DiscreteHamiltonian, SimulatedAnnealing
import numpy as np

model = EdwardsAnderson2D(L=16, disorder="pm1", seed=0)
Hd = DiscreteHamiltonian(model)
result = SimulatedAnnealing(Hd, beta_schedule=np.linspace(0.1, 2.0, 5000), seed=0).run(n_steps=5000)
print(result["summary"]["best_energy"])
```

Every optimizer and sampler returns the same shape:

```python
{"summary": {...},      # scalar metrics: final/best energy, acceptance, runtime
 "trace":   {...},      # downsampled 1d arrays: step, energy, grad_norm, ...
 "artifacts": {...}}    # final_state, best_state, optional samples
```

## Command-line interface

Named presets drive the canonical scaling studies end-to-end:

```bash
python -m spinglass canonical --list-presets
python -m spinglass canonical --preset ferro_scaling --out ./results
python -m spinglass canonical --preset ea_scaling
python -m spinglass canonical --preset er_glass
python -m spinglass canonical --preset sk
```

Each preset sweeps `beta`, averages over independent disorder realizations, and writes a timestamped run directory with one JSON per panel (`discrete_sampling`, `discrete_optimization`, `relaxed_sampling`, `relaxed_optimization`) plus per-condition overlap summaries. For one-off sweeps use `python -m spinglass beta-sweep --task sampling --model ea2d --L 8 --betas 0.3,0.8,1.5`.

## Implemented methods

**Discrete** — `GreedySpinDescent`, `SimulatedAnnealing`, `ParallelTemperingOptimizer`, `MetropolisSampler`, `GibbsSampler`, `ParallelTemperingSampler`.

**Relaxed** — `GradientDescentOptimizer`, `AdamOptimizer`, `LangevinSampler`, `MALASampler`, `HMCSampler`.

All discrete single-spin methods route through a cached local-field update (`DiscreteHamiltonian.column_cache()`) so the hot loop is O(k_i) per accepted flip on sparse models, not O(n).

## Experiments and diagnostics

- `run_single` / `run_grid` — single-run and sweep execution with flat metadata
- `summarize_optimization_table` / `summarize_sampling_table` — grouped benchmark rows
- `Budget(kind="sweeps", value=150)` + friends — matched-compute convention for fair discrete-vs-relaxed comparisons
- `sampling_beta_sweep`, `optimization_beta_sweep`, `canonical_study` (in `spinglass.experiments.studies`) — reusable study functions that back the CLI presets and accept `n_disorders` for instance averaging

Diagnostics: `acf`, `ess`, `rhat`, `integrated_autocorr_time`, `magnetization`, `overlap`, `pairwise_overlaps`, plus glassy helpers (`summarize_replica_overlaps`, `overlap_histogram`, `summarize_overlap_mixing`).

Plotting: trace / ACF / rank-histogram / pair / pair-matrix diagnostics, difficulty and grouped-metric plots, and overlap plots (`plot_overlap_histogram`, `plot_mean_abs_q_curve`, `plot_overlap_histograms_by_beta`, `plot_overlap_vs_energy`). Call `set_publication_style()` once at the top of a notebook for publication defaults.

## Repository layout

```text
spinglass/
  models/        model containers (disorder + topology)
  couplings/     coupling matrix builders
  hamiltonian/   DiscreteHamiltonian, RelaxedHamiltonian
  optimizers/    one file per algorithm
  samplers/      one file per algorithm
  diagnostics/   observables, MCMC stats, run summaries
  experiments/   builders, grids, runner, budget, studies, presets, benchmarks,
                 overlap, results_dir, io
  plotting/      style, traceplots, diagnostics, difficulty, overlap
  utils/         rng, records, spin (ColumnCache + incremental updates)
  cli.py         `python -m spinglass ...` entrypoint
tests/           51 unit tests + zero-dependency runner
scripts/         thin drivers for the canonical study
```

Design split: a **model** holds disorder and topology; a **Hamiltonian** performs the physics; **algorithms** operate on a Hamiltonian; **experiments** run algorithms repeatedly and collect analysis-ready summaries.

## Testing

```bash
python -m tests.run_all     # zero-dep runner — always works
pytest tests/               # if pytest is installed
```

51 tests cover energy vs brute force, incremental local-field updates, analytic gradient vs finite differences, PT swap bookkeeping, ESS / R-hat / tau_int on iid + AR(1) chains, Budget resolution, disorder fan-out, ColumnCache correctness plus speedup, and the preset registry.

## Core conventions

- `H(s) = -0.5 * s^T J s`, `J` symmetric with zero diagonal, sparse CSR or dense ndarray (use `J @ v` for both).
- Discrete states are `int8` in {-1, +1}; cast to float64 before matvecs.
- Relaxed surrogate is `t = tanh(alpha x)` with optional `lam` penalty; `RelaxedHamiltonian.project(x)` maps back to discrete via `sign(.)`.
- Samplers maintain `h = J @ s` incrementally through the column cache; don't recompute it in a hot loop.
- Preserve the `summary` / `trace` / `artifacts` return shape when adding new algorithms so the experiment and plotting layers work unchanged.
