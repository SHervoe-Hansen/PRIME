# PRIME

**Protonation-number Reweighting in Multi-pH Ensembles**

Free energies, titration curves and protonation-number distributions at any pH, from the protonation numbers recorded in constant-pH simulations.

PRIME is the multistate Bennett acceptance ratio (MBAR) specialized to the semi-grand ensemble of protons. The pH windows of a constant-pH simulation differ only through the term $\ln 10 \cdot \mathrm{pH} \cdot N$, so the potential energy cancels from the MBAR equations and the sum over configurations collapses onto the protonation number $N$. The histogram of $N$ in every window is therefore all that PRIME needs, and the compression is exact. The package is the reference implementation of

> S. Hervø-Hansen and N. Matubayasi, *Protonation-number Reweighting for pH-Dependent Free Energy Calculations*, submitted. DOI to follow.

The only input is $N(t)$ for every pH window, one integer per frame. No energies, no coordinates and no site assignments are needed, so the package works with the output of any constant-pH code.

## Installation

With conda, from a clone of this repository

```bash
conda env create -f environment.yml      # numpy, scipy, matplotlib, jupyterlab, pytest and the package itself
conda activate prime
```

With pip, into an existing environment

```bash
pip install .                            # the package and its two dependencies, numpy and scipy
pip install -e ".[notebooks,test]"       # editable, plus matplotlib, jupyterlab, pillow and pytest
```

Python 3.9 or newer is required. The distribution is named `prime-reweighting` and the module is imported as `prime`. An unrelated package named `prime` exists on PyPI, so install this one from the repository rather than by that name.

## Quick start

```python
import numpy as np
import prime

# records[k] is the integer array N(t) of window k, simulated at pHs[k]
H = prime.build_histograms(records, pHs, n_sites=n_sites)              # (K, n_sites + 1)
fit = prime.PRIME(H, pHs, n_sites=n_sites)

grid = np.linspace(2.0, 12.0, 101)
dY = fit.free_energy(grid, pH_ref=2.0)         # beta [Upsilon(pH) - Upsilon(2)]
mean, var = fit.mean(grid), fit.variance(grid) # titration curve and its slope
N, Pr = fit.distribution(7.0)                  # Pr(N | pH 7)
pKa = fit.stepwise_pka()                       # stepwise pKa of the steps N-1 -> N

# uncertainty from a block bootstrap of the histograms
h_blk = prime.block_histograms(records, pHs, n_blocks=32, n_sites=n_sites)
nominal, boots, lo, hi = prime.bootstrap(h_blk, pHs, prime.estimator_profiles, n_boot=1000)

# diagnostics
print(prime.format_consistency(fit.consistency(h_blk=h_blk)))   # do the windows agree on every pKa?
print(fit.connectivity()["n_components"])                        # 1 if the fit is identifiable
```

## What is in the package

| Section of the paper | Functions |
|---|---|
| Histograms and blocks | `build_histograms`, `block_histograms`, `partial_histograms`, `histogram_moments`, `block_means` |
| Correlation of the records | `statistical_inefficiency`, `block_length`, `blocking_scan` |
| The estimator | `solve_prime`, `PRIME` |
| Uncertainty | `compute_prime_errors`, `resample_histograms`, `bootstrap`, `estimator_profiles`, `transition_profiles` |
| Identifiability | `overlap_matrix`, `connectivity_diagnostics`, `support_connected` |
| Weights of the protonation numbers, anything at any pH | `log_weights`, `reweight`, `distribution`, `stepwise_pka_from_weights`, `edge_weights` |
| Thermodynamic integration, for reference | `compute_TI_trapezoid`, `compute_TI_hermite`, `fourth_cumulant`, `ti_bias_trapezoid`, `ti_bias_hermite` |
| Consistency test | `stepwise_pka`, `consistency_test`, `format_consistency` |
| Residual against TI | `residual_decomposition` |
| Fewer windows | `coarsening_rms`, `predict`, `variance_maxima` |

`help(prime.core)` gives the overview with the conventions, and every function carries a docstring with its equations.

The conventions in short. Free energies are reduced relative semi-grand potentials $\beta[\Upsilon(\mathrm{pH}) - \Upsilon(\mathrm{pH}_\mathrm{ref})]$. Histograms have one row per window and one column per protonation number from 0 to the number of sites, which is why `n_sites` is passed, so that the physical bounds of $N$ are known even where they were never sampled. All Boltzmann sums are evaluated in log space, so systems with thousands of titratable sites do not overflow. Uncertainties come from a block bootstrap of the histograms, with blocks longer than the correlation time of $N(t)$ given by `block_length`.

## Tests

```bash
pytest
```

The 25 tests take a few seconds. They check the estimator against the analytic free energy of independent sites, the bootstrap against repeated simulations, the biases of the two thermodynamic-integration rules against the Euler–Maclaurin expressions, and the prediction rule on records with a known answer.

## Figure reproduction & example package usage

[`PRIME_paper.ipynb`](PRIME_paper.ipynb) reproduces the figures and tables of the paper from the protonation records of the insulin simulations, which are deposited separately (Zenodo, DOI to follow). The deposit holds one directory per condition, `Simulations/[condition]/5_analysis`. Unpack the records as described below and set the path to the `Simulations` directory at the top of the notebook.

## Unpacking the protonation records

The `reordered_cpouts.pH_*` files are stored in packed form, one `.protpack` file per window. Before analysing the data, restore them by running the script in the directory of each condition the notebook uses:

```bash
cd Simulations/[condition]/5_analysis
python protpack.py
```

The plain files appear next to the packed ones with their original names, and each is checked against its sha256 sum. This needs Python 3 and numpy and takes a few minutes per condition. Running the script again does nothing once both forms are present. The analysis notebook reads the restored files.

## Citation

If you use PRIME, please cite the paper above. The citation of the package itself will be added with its DOI.

## License

MIT, see [`LICENSE`](LICENSE).
