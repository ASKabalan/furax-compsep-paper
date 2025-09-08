# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the research code and paper for **FURAX CMB Component Separation Framework** - a JAX-powered framework for optimizing parametric component separation in Cosmic Microwave Background (CMB) polarization data analysis. The main innovation is using spherical K-means clustering with variance-based model selection to adaptively partition the sky for different spectral parameters, improving detection of primordial gravitational waves.

## Quick Start

1. **Setup environment**: `pip install -r requirements.txt`
2. **Validate installation**: `python -c "import jax; print(jax.devices())"` and `python -c "import furax"`
3. **Run quick test**: `cd content && python 02-validation-model.py -n 64 -m GAL020`
4. **Check code quality**: `ruff check . && ruff format .`

## Commands

### Package Installation and Setup
```bash
pip install -r requirements.txt    # Install all dependencies (includes FURAX from GitHub)
export EQX_ON_ERROR=nan            # Enable JAX debugging (recommended for development)

# Validate installation
python -c "import jax; print('JAX devices:', jax.devices())"  # Check JAX/GPU setup
python -c "import furax; print('FURAX package installed successfully')"  # Verify package
```

### Code Quality and Linting
```bash
ruff check .          # Lint Python code with ruff (configured in ruff.toml)
ruff format .         # Format Python code with ruff
ruff check --fix .    # Auto-fix linting issues where possible
```

### Paper and Documentation
```bash
# Compile LaTeX paper (from publications/paper/ directory)
cd publications/paper
pdflatex furax_comp_sep.tex    # Compile paper to PDF
bibtex furax_comp_sep          # Process bibliography 
pdflatex furax_comp_sep.tex    # Final compilation with references
```

### Testing and Validation
```bash
# Run code quality checks
ruff check .                   # Check for linting issues
ruff format .                  # Format code automatically

# Basic functionality validation
python -c "import jax; print('JAX devices:', jax.devices())"  # Check JAX/GPU setup
python -c "import furax; print('FURAX imported successfully')"  # Verify FURAX package

# Run individual experiments for testing (from content/ directory)
cd content
python 02-validation-model.py -n 64 -m GAL020    # Quick validation run (nside=64, GAL020 mask)
python 01-bench_bcp.py -n 64 -s -l              # Benchmark component separation (with stats and logging)
python 03-noise-model.py -n 64 -ns 20 -nr 0.2 -m GAL020  # Test with 20% noise ratio
```

### Running Experiments

#### Individual Scripts (run from content/ directory)
```bash
# Benchmarking scripts
python 01-bench_bcp.py -n 64 -s -l                    # Benchmark basic component separation
    # -n: HEALPix nside resolution, -s: show stats, -l: enable logging
python 01-bench_clusters.py -n 64 -cl 100             # Benchmark clustering performance  
    # -cl: number of clusters to test

# Model validation
python 02-validation-model.py -n 64 -m GAL020         # Run validation model
    # -n: nside, -m: mask type (GAL020/040/060, GALACTIC)
python 03-noise-model.py -n 64 -ns 100 -nr 1.0 -m GAL020  # Noise model testing
    # -ns: number of samples, -nr: noise ratio (0.0-1.0)

# Main K-means clustering method (key innovation)
python 08-KMeans-model.py -n 64 -pc 100 5 1 -tag c1d1s1 -m GAL020 -i LiteBIRD
    # -n: nside, -pc: clusters for [dust_temp, dust_beta, sync_beta]
    # -tag: simulation config, -m: mask, -i: instrument

# Tensor-to-scalar ratio estimation
python 09-R_estimation.py                             # Estimate r parameter

# Other methods for comparison
python 05-PTEP-model.py -n 64 -ud 64 0 2 -tag c1d1s1 -m GAL020 -i LiteBIRD  # Multi-resolution
python 07-single_patch.py -n 64 -tag c1d1s1 -m GAL020 -i LiteBIRD           # Single patch
```

#### Full Experiment Suite (SLURM)
```bash
cd content
./99-run_all_jobs.sh    # Submit all SLURM jobs for complete analysis
```

### Data Management and Troubleshooting
```bash
# Cache management (frequency maps cached to avoid regeneration)
ls content/freq_maps_cache/                           # View cached frequency maps
rm content/freq_maps_cache/freq_maps_nside_*.pkl      # Clear cache if regeneration needed

# GPU and JAX troubleshooting
python -c "import jax; print('JAX devices:', jax.devices())"  # Check GPU availability
nvidia-smi                                            # Check GPU status (SLURM environments)
export EQX_ON_ERROR=nan                              # Enable JAX NaN debugging
export JAX_TRACEBACK_FILTERING=off                   # Full JAX tracebacks

# Results and plots
ls results/                                          # View experiment results
ls content/plots/                                    # View generated plots
# Results use naming: {method}_{config}_{instrument}_{mask}_{samples}/
```


## Repository Structure

### Core Research Code (`content/`)
- `01-bench_*.py`: Performance benchmarking scripts
- `02-validation-model.py`: Model validation framework  
- `03-noise-model.py`: Noise modeling and analysis
- `04-distributed-gridding.py`: Distributed grid search optimization
- `05-PTEP-model.py`: Multi-resolution baseline model (LiteBIRD PTEP)
- `06-gibbs.py`: Gibbs sampling implementation
- `07-single_patch.py`: Single global patch model
- `08-KMeans-model.py`: **Main contribution** - Adaptive K-means clustering
- `09-R_estimation.py`: Tensor-to-scalar ratio estimation
- `99-run_all_jobs.sh`: SLURM job submission script
- `99-slurm_runner.slurm`: SLURM execution template

### Data Infrastructure (`data/`)
- `generate_maps.py`: CMB and foreground map simulation using PySM3/FGBuster
- `plotting.py`: Visualization and analysis utilities
- `instruments.py`/`instruments.yaml`: Instrument specifications (LiteBIRD, etc.)
- `masks/`: Galactic masks at various resolutions

### Package Source Code (not in this repo)
The FURAX package (`furax_cs`) is installed as an external dependency. Core functionality includes:
- Component separation algorithms (K-means, multi-res, single patch)
- Data handling utilities (instruments, masks, simulation)
- Tensor-to-scalar ratio estimation pipeline  
- Common utilities (caching, I/O)
- Visualization and benchmarking tools

### Results Storage (`results/`)
Structured naming: `{method}_{config}_{instrument}_{mask}_{samples}/`
- `best_params.npz`: Optimized spectral parameters
- `results.npz`: Full analysis results
- `mask.npy`: Sky mask used

## Technical Architecture

### Core Framework (FURAX)
- **JAX-native**: End-to-end differentiability with JIT compilation and GPU acceleration
- **PyTree Integration**: All data structures (configs, results) are JAX PyTrees for efficient transformations
- **Modular operators**: Mixing matrix, noise weighting, parameter dispatch as composable linear operators
- **Memory efficient**: Uses linear operators instead of explicit matrices to handle large HEALPix maps
- **Scalable**: Vectorized execution across simulation ensembles with `vmap` and distributed computing

### Component Separation Model
The parametric component separation follows the fundamental equation:
```
d = A(β)s + n
```
Where:
- `d`: Observed frequency maps (data)
- `A(β)`: Mixing matrix dependent on spectral parameters β
- `s`: Sky signal components (CMB, dust, synchrotron)  
- `n`: Instrumental noise
- `β`: Spatially-varying spectral parameters (dust temperature/beta, synchrotron beta)

### Key Innovation: Adaptive K-means Clustering
1. **Spherical K-means**: Partitions sky using RA/Dec coordinates with 3D Cartesian averaging
2. **Adaptive Parameter Assignment**: Different spectral parameters β per cluster
3. **Variance-based Selection**: Minimizes CMB reconstruction variance as proxy for foreground contamination
4. **Grid Search Optimization**: Exhaustive search over clustering configurations (number of clusters per parameter)

### Distributed Execution Model
- **SLURM Integration**: Designed for GPU clusters using SLURM job scheduler
- **MPI-style Distribution**: Uses JAX distributed computing with fault tolerance
- **Multi-GPU Support**: Scales across multiple A100 GPUs on Jean Zay cluster
- **Caching Strategy**: Frequency maps cached to avoid redundant PySM3 simulations
- **Batch Processing**: Supports ensemble simulations with different noise realizations

### Dependencies and Package Architecture
Core dependencies from `requirements.txt` and `setup.py`:

**JAX Ecosystem:**
- `furax`: Core JAX-powered component separation framework (git+https://github.com/CMBSciPol/furax)
- `jax-healpy`: HEALPix operations in JAX for sky map processing
- `jax-grid-search`: Distributed optimization framework (git+https://github.com/ASKabalan/jax-grid-search)
- `jaxopt`: L-BFGS optimization for parameter fitting

**Component Separation:**
- `fgbuster`: Baseline component separation for comparison (git+https://github.com/fgbuster/fgbuster@clusters)
- `camb`: CMB power spectrum computation for r estimation

**Development Tools:**
- `jax_hpc_profiler`: JAX performance profiling for GPU optimization
- `pre-commit`: Code quality hooks with ruff integration
- `ipython`, `ipykernel`, `ipywidgets`: Notebook development support

### HEALPix and Sky Conventions
- **nside parameter**: HEALPix resolution where npix = 12×nside² (e.g., nside=64 → 49,152 pixels)
- **Galactic masks**: GAL020/040/060 refer to masks excluding 20%/40%/60% of galactic contamination
- **Coordinates**: Uses RA/Dec coordinates converted to 3D Cartesian for spherical K-means
- **Stokes parameters**: Supports I, Q, U polarization with focus on B-mode CMB detection

## Development Notes

### Code Style
- Python 3.11+ target (configured in `ruff.toml`)
- Line length: 100 characters
- Use ruff for linting and formatting
- Include type hints for new functions

### Performance Considerations
- Use JAX transformations (`jit`, `vmap`, `pmap`) for performance-critical code
- **Frequency map caching**: Maps cached in `content/freq_maps_cache/` with naming pattern `freq_maps_nside_{nside}_{noise}_{config}.pkl`
- **Cache key components**: nside resolution, noise settings (noise/no_noise), sky configuration tags
- Prefer linear operators over explicit matrices for memory efficiency
- Use `EQX_ON_ERROR=nan` environment variable for debugging JAX operations

### Computational Environment
- Designed for GPU clusters (tested on Jean Zay with A100s)
- Uses MPI-style distributed execution with fault tolerance
- SLURM job scripts configured for various GPU configurations

### File Organization
- Experiments follow numerical naming scheme (`01-`, `02-`, etc.)
- Results use structured naming for automatic analysis: `{method}_{config}_{instrument}_{mask}_{samples}/`
- Plots are versioned and organized by method in `content/plots/`
- Cache files prevent redundant computations in `content/freq_maps_cache/`
- All scripts should be run from the `content/` directory
- SLURM jobs are configured for Jean Zay cluster with A100 GPUs

## Development Workflow

1. **Set up environment**: Install dependencies with `pip install -r requirements.txt`
2. **Run experiments**: Use `99-run_all_jobs.sh` or individual scripts from `content/` directory
3. **Generate plots**: Results automatically saved to `content/plots/`
4. **Analyze results**: Use notebooks in `notebooks/` for analysis and visualization
5. **Validate code**: Run `ruff check .` and `ruff format .` before committing

## Common Issues and Solutions

### JAX/GPU Issues
- Set `export EQX_ON_ERROR=nan` for better JAX debugging
- Use `jax.devices()` to check GPU availability
- Check SLURM GPU allocation with `nvidia-smi`

### Missing Dependencies
- All custom packages are git repositories - ensure you have access to GitHub
- Use `pip install -r requirements.txt` to install all dependencies
- Some packages require specific branches (e.g., `fgbuster@clusters`)

### Cache Management
- **Location**: Frequency maps cached in `content/freq_maps_cache/` to avoid expensive PySM3 regeneration
- **Naming pattern**: `freq_maps_nside_{nside}_{noise_setting}_{config_tag}.pkl`
  - Example: `freq_maps_nside_64_noise_c1d1s1.pkl`
- **Cache keys**: nside resolution, noise vs no_noise, configuration tag (c1d1s1, etc.)
- **Clear cache**: Delete specific `.pkl` files to force regeneration with different parameters
- **Storage**: Cache files can be large (~GB) - monitor disk usage on clusters

## Notebooks

### Tutorial Notebooks (`notebooks/`)
- `00_Tutorial_Overview.ipynb`: Master tutorial overview and learning path
- `01_FGBuster_vs_FURAX_Comparison.ipynb`: Framework comparison and performance benchmarking
- `02_KMeans_Adaptive_Component_Separation.ipynb`: **Main innovation** - Adaptive K-means clustering tutorial
- `03_Multi_Resolution_Component_Separation.ipynb`: PTEP baseline and hierarchical approaches
- `04_Tensor_Scalar_Ratio_Estimation.ipynb`: Complete r estimation pipeline with CAMB integration

### Analysis Notebooks (`content/`)
- `80-illustraions.ipynb`: Figure generation for paper (**Note**: filename contains typo in codebase)
- `80-plot_runs.ipynb`: Results plotting and analysis

## Notebook Features
- **Google Colab compatible**: All notebooks include Colab badges for cloud execution
- **Progressive difficulty**: Beginner → Intermediate → Advanced progression
- **Publication-quality visuals**: Professional plots suitable for presentations
- **Interactive learning**: Step-by-step explanations with executable code
- **Cross-referenced**: Links between notebooks and paper sections