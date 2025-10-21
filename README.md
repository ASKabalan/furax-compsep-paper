# FURAX Component Separation Paper

A research project comparing CMB (Cosmic Microwave Background) component separation frameworks, specifically **FURAX** (JAX-based) versus **FGBuster** (traditional NumPy-based). The project focuses on separating CMB signals from galactic foregrounds (dust and synchrotron emission) using modern high-performance computing techniques.

## Features

- **FURAX**: Modern JAX-based framework with automatic differentiation, GPU acceleration, and JIT compilation
- **FGBuster**: Traditional NumPy-based component separation library (baseline comparison)
- **Cross-validation**: All methods validated to ensure consistent results
- **Advanced Methods**: K-means adaptive separation, multi-resolution (PTEP) approaches
- **Comprehensive Analysis**: R-statistic estimation, residual analysis, variance optimization

## Installation

### 1. Install JAX

JAX installation depends on your hardware configuration. Follow the official JAX installation guide:

**CPU-only installation:**
```bash
pip install jax
```

**GPU installation (CUDA):**
```bash
pip install jax[cuda]
```

For more details, see the [JAX installation documentation](https://github.com/google/jax#installation).

### 2. Install furax-cs Package

Install the package in development mode:

```bash
pip install furax-cs
```

This will install all dependencies and register the command-line scripts.

## Quick Start

### Generate Cached Data

Before running experiments, generate cached frequency maps:

```bash
# Generate maps for common resolutions
generate_data --nside 64 128 --sky c1d0s0 c1d1s1

# Generate galactic masks
generate_masks --nside 64 128 256 512
```

### Run Simple Benchmarks Locally

```bash
# Basic framework comparison
bench-bcp -n 64 -s -l

# Validation without noise
validation-model -n 64 -m GAL020 -mi 1000

# K-means adaptive method
kmeans-model -n 64 -pc 100 10 1 -tag c1d1s1 -m GAL020 -i LiteBIRD -mi 1000

# Multi-resolution PTEP method
ptep-model -n 64 -ud 64 32 16 -tag c1d1s1 -m GAL020 -i LiteBIRD -mi 1000
```

## Available Scripts

The package provides the following command-line entry points:

| Script | Entry Point | Description |
|--------|-------------|-------------|
| Benchmarking | `bench-bcp` | Basic component separation benchmarks (FGBuster vs FURAX, TNC/L-BFGS solvers) |
| | `bench-clusters` | K-means clustering performance analysis |
| Validation | `validation-model` | Framework validation without noise |
| | `noise-model` | Noise robustness testing (supports `-ns` and `-nr` flags) |
| Advanced Methods | `distributed-gridding` | Distributed grid search (multi-node, multi-GPU) |
| | `ptep-model` | Multi-resolution component separation with `-ud` parameters |
| | `single-patch` | Single patch analysis |
| | `kmeans-model` | K-means adaptive separation with `-pc` parameters |
| Analysis | `r_analysis` | R-statistic analysis and visualization pipeline |
| Data Generation | `generate_data` | Generate and cache frequency maps |
| | `generate_masks` | Downgrade Planck galactic masks to target resolutions |

### Common Arguments

- `-n, --nside`: HEALPix resolution (typically 32, 64, 128, 256, 512, 1024)
- `-m, --mask`: Sky mask (GAL020, GAL040, GAL060, GAL020_U, GAL020_L, GALACTIC)
- `-i, --instrument`: Instrument name (LiteBIRD, SO_SAT, SO_LAT, Planck)
- `-ns, --n_sims`: Number of noise simulations
- `-nr, --noise_ratio`: Noise level ratio (0.2 = 20%, 1.0 = 100%)
- `-cl, --n_clusters`: Number of K-means clusters
- `-tag`: Sky model tag (e.g., c1d1s1 = CMB + dust + synchrotron)
- `-ud`: Multi-resolution downgrade parameters for PTEP model
- `-pc`: Patch configuration for K-means model
- `-mi, --max-iter`: Maximum L-BFGS solver iterations

## R-Analysis Pipeline

The `r_analysis` tool provides comprehensive post-processing and visualization:

### Basic Usage

```bash
# Analyze results with default settings
r_analysis -r "compsep_c1d1s1_LiteBIRD_GAL020" -t "My Run" -n 64 -i LiteBIRD

# Plot all analysis outputs
r_analysis -r "compsep_*_GAL020" -t "Full Analysis" -a

# Cache expensive computations only
r_analysis -r "compsep_*" -t "Cache Run" --cache-only

# Compare multiple runs
r_analysis -r "compsep_*_GAL020" "compsep_*_GAL040" \
           -t "GAL020 Run" "GAL040 Run" \
           -pi -arc -arv
```

### Plotting Options

- `-pp`: Plot spectral parameter maps (β_d, T_d, β_s)
- `-pt`: Plot patch assignments
- `-pc`: Plot CMB reconstructions
- `-ps`: Plot power spectra (individual runs)
- `-pr`: Plot r-statistic estimates (individual runs)
- `-as`: Plot all power spectra (comparison)
- `-ar`: Plot all r-statistic estimates (comparison)
- `-arc`: Plot r vs. number of clusters
- `-arv`: Plot r vs. variance
- `-pi`: Plot illustrations (combined metrics)
- `-a`: Plot everything

For full documentation on r_analysis, run:
```bash
r_analysis --help
```

## Notebooks

Jupyter notebooks with detailed analysis and visualizations are available in the `notebooks/` directory:

- **01_FGBuster_vs_FURAX_Comparison.ipynb**: Framework validation and comparison
- **02_KMeans_Adaptive_Component_Separation.ipynb**: K-means method analysis
- **03_PTEP_Multi_Resolution_Component_Separation.ipynb**: Multi-resolution approach analysis
- **04_Scripts_and_Analysis_Workflow.ipynb**: Complete workflow demonstration
- **80-illustraions.ipynb**: Visual illustrations for publications
- **80-plot_runs.ipynb**: Benchmark result analysis and plots

See [notebooks/README.md](notebooks/README.md) for more details.

## HPC Usage (Jean Zay Supercomputer)

For large-scale experiments on HPC systems:

```bash
# Individual job submission
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 \
       slurms/slurm_runner.slurm bench-bcp -n 64

# Run complete benchmark suite (all experiments)
cd slurms/
bash run_all_jobs.sh
```

The SLURM scripts automatically:
- Detect GPU partition (gpu_p5/A100, gpu_p6/H100, or default/V100)
- Load partition-specific virtual environments
- Organize outputs by GPU type, node count, script name, and arguments

### Output Organization

- **Standard outputs**: `traces/{gpu_name}/{nb_gpus}/{script_name}/{args}/`
- **Profiling outputs**: `prof_traces/` and `out_prof/` (same structure)
- **Results**: `results/compsep_{tag}_{instrument}_{mask}_{noise}/`

## Project Structure

```
furax-compsep-paper/
├── src/furax_cs/               # Main package
│   ├── data/                   # Data generation and management
│   │   ├── instruments.yaml    # Instrument configurations
│   │   ├── generate_maps.py    # Map generation and caching
│   │   ├── plotting.py         # Visualization utilities
│   │   └── masks/              # Galactic mask utilities
│   ├── r_analysis/             # R-statistic analysis pipeline
│   │   ├── main.py             # Main analysis orchestration
│   │   ├── r_estimate.py       # R-statistic estimation
│   │   ├── residuals.py        # Residual computation
│   │   ├── plotting.py         # Analysis visualizations
│   │   └── caching.py          # Result caching system
│   └── scripts/                # Analysis scripts (entry points)
├── notebooks/                  # Jupyter analysis notebooks
├── slurms/                     # HPC job scripts
│   ├── slurm_runner.slurm      # Job wrapper with profiling
│   └── run_all_jobs.sh         # Complete benchmark suite
├── pyproject.toml              # Package configuration
└── CLAUDE.md                   # Claude Code assistant instructions
```

## Code Quality

Lint and format code with ruff:

```bash
# Check for issues
ruff check

# Auto-format code
ruff format
```
## Development Workflow

1. **Setup**: `pip install -e .`
2. **Development**: Test locally with small NSIDE values (32, 64)
3. **Cache generation**: Run `generate_data` to create frequency maps
4. **Validation**: Compare FURAX and FGBuster results for consistency
5. **Scaling**: Submit HPC jobs for production runs with larger NSIDE
6. **Analysis**: Use notebooks or `r_analysis` for visualization
