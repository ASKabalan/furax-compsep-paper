# r_analysis Package

A modular package for R estimation, CMB component separation analysis, and result validation. This package powers the `r_analysis` CLI tool.

## Package Structure

```
r_analysis/
├── __init__.py          # Package initialization & exports
├── main.py              # Main entry point (orchestrates snap, plot, validate, estimate)
├── parser.py            # Command-line argument parsing (subcommand based)
├── utils.py             # Utility functions (expand_stokes, params_to_maps)
├── caching.py           # Caching management for expensive computations
├── residuals.py         # Residual computation (systematic, statistical, total)
├── r_estimate.py        # R estimation (likelihood, CAMB templates)
├── plotting.py          # All plotting functions
├── compute.py           # Core computation flags and logic
├── snapshot.py          # Snapshot management (save/load)
├── validate.py          # NLL Validation logic
└── run_grep.py          # Run filtering and regex matching
```

## Usage

### Command-line Interface

The `r_analysis` tool uses a subcommand structure: `snap`, `plot`, `validate`, and `estimate`.
**Important**: You must specify the subcommand *before* any other arguments.

#### 1. Snapshot (`snap`)
Compute statistics and save them to a lightweight `.npz` file. Useful for aggregating heavy results from HPC runs.

```bash
r_analysis snap -n 64 -r "kmeans" -ird results/ -o snapshots/my_run.npz
```

#### 2. Plotting (`plot`)
Generate plots from raw results or a pre-computed snapshot.

```bash
# Plot everything from a snapshot
r_analysis plot -n 64 -r ".*" -ird results/ --snapshot snapshots/my_run.npz -a

# Plot specific metrics (e.g., r vs clusters)
r_analysis plot -n 64 -r "kmeans" -ird results/ -arc
```

#### 3. Validation (`validate`)
Run profile likelihood validation on the results.

```bash
r_analysis validate -n 64 -r "kmeans" -ird results/ --steps 5 --noise-ratio 0.0
```

#### 4. Estimation (`estimate`)
Estimate the tensor-to-scalar ratio $r$ directly from a spectrum or map file.

```bash
r_analysis estimate --cmb cmb_spectrum.npy --fsky 0.8
```

### Python API

You can import and use the package programmatically:

```python
from furax_cs.r_analysis import run_analysis, estimate_r

# Run the full analysis workflow (parses sys.argv)
run_analysis()

# Or use specific components
r_best, sigma_neg, sigma_pos, r_grid, L_vals = estimate_r(
    cl_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_noise, f_sky
)
```

## Key Features

- **Subcommand Architecture**: Clean separation of compute, visualize, and validate modes.
- **Snapshot System**: Decouples expensive computation from interactive plotting.
- **Regex Filtering**: Flexible `-r` argument to select runs by pattern.
- **Modular Design**: Each module (`residuals`, `plotting`, `caching`) has a single responsibility.
- **Comprehensive Metrics**:
    - **Systematic Residuals**: Bias in component separation.
    - **Statistical Residuals**: Noise and variance.
    - **R Estimation**: 1D likelihood for primordial B-modes.

## Command-line Arguments

See `parser.py` or run `r_analysis --help` for the full list.

### Global Arguments (per subcommand)
These arguments are available for `snap`, `plot`, and `validate`.

*   `-n`, `--nside`: HEALPix resolution.
*   `-i`, `--instrument`: Instrument configuration (`LiteBIRD`, `Planck`, `default`).
*   `-r`, `--runs`: Regex patterns to filter run directories.
*   `-ird`, `--input-results-dir`: Directory containing results.

### Plotting Toggles (for `plot`)
*   `-a`, `--plot-all`: Enable all plots.
*   `-arc`: Plot $r$ vs number of clusters.
*   `-as`: Plot all power spectra.
*   `-ac`: Plot all CMB reconstructions.
*   `-psm`: Plot systematic residual maps (single run).
*   `-pr`: Plot $r$ likelihood (single run).

### Validation Arguments (for `validate`)
*   `--steps`: Number of steps for profile likelihood.
*   `--noise-ratio`: Noise level for validation.
*   `--scales`: Scales for perturbation.
