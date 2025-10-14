# r_analysis Package

A modular package for R estimation and CMB component separation analysis, refactored from `09-R_estimation.py`.

## Package Structure

```
r_analysis/
├── __init__.py          # Package initialization
├── main.py              # Main workflow (compute_results, plot_results, run_analysis)
├── parser.py            # Command-line argument parsing
├── utils.py             # Utility functions (expand_stokes, params_to_maps, etc.)
├── caching.py           # Caching and snapshot management
├── residuals.py         # Residual computation (systematic, statistical, total)
├── r_estimate.py        # R estimation (likelihood, CAMB templates)
├── plotting.py          # All plotting functions
└── run_management.py    # Run filtering and specification parsing
```

## Usage

### Command-line Interface

Run the analysis using the entry script:

```bash
cd content/
python 10-R_estimation_v2.py -n 64 -i LiteBIRD -r "run_name" -t "Run Title" -a
```

### Python API

Import and use specific functions:

```python
from r_analysis import run_analysis, parse_args, estimate_r

# Run full analysis workflow
run_analysis()

# Or use individual components
args = parse_args()
r_best, sigma_neg, sigma_pos, r_grid, L_vals = estimate_r(
    cl_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_noise, f_sky
)
```

## Key Features

- **Modular design**: Each module has a single responsibility
- **Caching support**: Expensive computations (W_D_FG) are cached and reused
- **Snapshot system**: Incremental plotting with saved intermediate results
- **Run management**: Flexible filtering and batch processing
- **Comprehensive plotting**: Individual and comparison plots for all analyses

## Migration from 09-R_estimation.py

The original `content/09-R_estimation.py` remains **untouched** for backward compatibility.

The new `content/10-R_estimation_v2.py` script uses this package and provides identical functionality with better organization.

## Command-line Arguments

See `parser.py` for the full list of options. Highlights:

**General configuration**
- `-n, --nside` – HEALPix resolution.
- `-i, --instrument` – Instrument (`LiteBIRD`, `Planck`, `default`).
- `-r, --runs` / `-t, --titles` – Filter runs and provide display titles.
- `-co, --cache-only` – Populate W·d_fg caches and exit.
- `-cr, --compute-residuals` – Force residual computations (`all`, `total`, `statistical`, `systematic`, `none`).
- `--snapshot` – Directory used to persist intermediate payloads.

**Single-run plots**
- `-pp, --plot-params` – Spectral parameter maps (β_d, T_d, β_s).
- `-pt, --plot-patches` – Patch assignment maps per parameter.
- `-pv, --plot-validation-curves` – Optimiser diagnostics.
- `-pc, --plot-cmb-recon` – Reconstructed vs. true CMB Q/U.
- `-psm, --plot-systematic-maps` – Systematic residual maps.
- `-ptm, --plot-statistical-maps` – Statistical residual maps.
- `-ps, --plot-cl-spectra` – Detailed BB power spectra.
- `-pr, --plot-r-estimation` – One-dimensional likelihood for r.

**Aggregated plots**
- `-arc, --plot-r-vs-c` – r + σ(r) vs. cluster count.
- `-avc, --plot-v-vs-c` – Variance vs. cluster count.
- `-arv, --plot-r-vs-v` – r vs. variance.
- `-ac, --plot-all-cmb-recon` – Mosaic of CMB residual maps.
- `-as, --plot-all-spectra` – Overlay of BB spectra.
- `-asm, --plot-all-systematic-maps` – Mosaic of systematic residuals.
- `-atm, --plot-all-statistical-maps` – Mosaic of statistical residuals.
- `-ar, --plot-all-r-estimation` – Overlay of r likelihood curves.
- `-am, --plot-all-metrics` – Histograms of variance, NLL, and ΣCℓ.

**Bundles**
- `-pi, --plot-illustrations` – Convenience bundle (parameters, patches, grid diagnostics).
- `-a, --plot-all` – Enable every plot and prerequisite computation.
