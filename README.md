# FURAX Component Separation

[![PyPI version](https://badge.fury.io/py/furax-cs.svg)](https://badge.fury.io/py/furax-cs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**FURAX-CS** (FURAX Component Separation) is a Python package designed to benchmark and implement advanced component separation techniques for Cosmic Microwave Background (CMB) analysis. It leverages **JAX** for high-performance computing on GPUs and implements novel adaptive clustering methods.

This project specifically focuses on comparing:
*   **FGBuster**: parametric component separation (standard).
*   **FURAX**: Adaptive, gradient-based separation with spatially varying spectral parameters.

---

## Installation

### 1. Prerequisites (JAX)
This package depends on JAX. To enable GPU acceleration (highly recommended), you must install the CUDA version of JAX **before** installing this package.

**For NVIDIA GPUs:**
```bash
pip install -U "jax[cuda]"
```

**For CPU only:**
```bash
pip install jax
```

### 2. Install Package
Clone the repository and install in editable mode:

```bash
pip install furax-cs[all] # or choose specific extras: plotting, benchmarks
```
---

## Quick Start

### Python API
You can load generated data and masks directly in your scripts:

```python
from furax_cs.data import load_from_cache, get_mask, save_to_cache

# Load frequency maps (automatically generates if missing)
# Returns: frequencies (Hz), maps (shape: [freqs, 3, npix])
save_to_cache(nside=64, sky="c1d1s1")
nu, freq_maps = load_from_cache(nside=64, sky="c1d1s1")

# Load a Mask (e.g., 59% sky  for example all except the galactic plane)
mask = get_mask("ALL-GALACTIC", nside=64)
```

---

## Full Workflow Example

Here is a complete pipeline from data generation to analysis.

### Step 1: Generate Data
First, create the simulated frequency maps (CMB + Dust + Synchrotron) and cache them.
This is equivalent to running save_to_cache in Python.

```bash
# Generate maps at Nside 64
generate_data --nside 64 --sky c1d1s1
```

The tag can be any Pysm configuration (e.g., `c1d1s1` for 1 CMB, 1 Dust, 1 Synchrotron component).

Or you can have custom CMB maps with different r values for example:

```bash
generate_data --nside 64 --sky cr3d1s1
```

This will generate a CMB map with r=0.003.

### Step 2: Run Component Separation
Run the adaptive K-means clustering model. This divides the sky into regions and optimizes spectral parameters for each region.

```bash
# Run K-means model with:
# - 100 clusters for Dust Beta
# - 10 clusters for Dust Temp
# - 1 cluster for Synchrotron Beta
kmeans-model -n 64 -pc 100 10 1 -m GAL020 -tag c1d1s1
```

### Step 3: Analyze Results
Use the analysis tool to aggregate results, compute statistics, and plot metrics.

```bash
# 1. Compute statistics (snapshot) for all runs matching "kmeans"
r_analysis snap -n 64 -r "kmeans" -ird results/ -o snapshots -mi 2000 -s active_set

# 2. Plot results from the snapshot
r_analysis plot -n 64 -r "kmeans" -ird results/ --snapshot snapshots -as -arc

# 3. (Optional) Estimate r from specific maps or spectra
r_analysis estimate --cmb cmb_spectrum.npy --fsky 0.8
```

---

## CLI Reference

### 1. `generate_data`
Pre-generates and caches frequency maps to speed up subsequent runs.
*   **Usage**: `generate_data --nside <N> --sky <TAG>`
*   **Example**: `generate_data --nside 128 --sky c1d1s1`

### 2. `kmeans-model`
The core adaptive separation tool using spherical K-means clustering.
*   **Arguments**:
    *   `-n`: Healpix Nside.
    *   `-pc`: Patch counts (clusters) for `[beta_dust, temp_dust, beta_synch]`.
    *   `-m`: Mask name (e.g., `GAL020`, `GAL040`, `GAL060`).
    *   `-ns`: Number of noise simulations (default 1).
*   **Example**:
    ```bash
    kmeans-model -n 64 -pc 50 10 10 -m GAL040 -nr 0.1
    ```

### 3. `ptep-model`
Multi-resolution component separation (Patch per pixel at different resolutions).
*   **Arguments**:
    *   `-ud`: Target Nside for `[beta_dust, temp_dust, beta_synch]` (downgrading resolution).
*   **Example**:
    ```bash
    # Dust params at Nside 32, Synch at Nside 16
    ptep-model -n 64 -ud 32 32 16
    ```

### 4. `r_analysis`
Comprehensive analysis and plotting suite.

#### The `-r` (Runs) Argument
The `-r` argument accepts a list of **regex patterns**. The tool filters result directories in `-ird` (input results dir) that match **ALL** provided patterns.

*   `r_analysis ... -r "kmeans" "GAL020"` -> Matches `results/kmeans_GAL020_...`
*   `r_analysis ... -r ".*"` -> Matches everything.

#### Subcommands
*   `snap`: Aggregates heavy data from many result files into a single lightweight `.npz` snapshot.
*   `plot`: Generates plots from raw results or a snapshot.
    *   `-as`: Plot all spectra.
    *   `-arc`: Plot Tensor-to-scalar ratio (r) vs number of clusters.
    *   `-ac`: Plot CMB reconstructions.
*   `validate`: Runs profile likelihood validation.
*   `estimate`: Estimate tensor-to-scalar ratio `r` from spectra or maps.
    *   `--cmb`: Path to CMB spectrum or map.
    *   `--fsky`: Sky fraction.

---

## Development

### Running Tests
```bash
pytest
```

### Pre-commit Hooks
Ensure code quality before committing:
```bash
pre-commit install
pre-commit run --all-files
```
