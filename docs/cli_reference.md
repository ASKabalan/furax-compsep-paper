# CLI Reference & Workflow

This guide covers the command-line interface (CLI) for the `furax-cs` package, including the full data generation and component separation workflow.

## Full Workflow Example

Here is a complete pipeline from data generation to analysis.

### Step 1: Generate Data
First, create the simulated frequency maps (CMB + Dust + Synchrotron) and cache them.

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
*For detailed usage of `r_analysis`, see [r_analysis.md](r_analysis.md).*

```bash
# 1. Compute statistics (snapshot) for all runs matching "kmeans"
r_analysis snap -n 64 -r "kmeans" -ird results/ -o snapshots -mi 2000 -s active_set

# 2. Plot results from the snapshot
r_analysis plot -n 64 -r "kmeans" -ird results/ --snapshot snapshots -as -arc

# 3. (Optional) Estimate r from specific maps or spectra
r_analysis estimate --cmb cmb_spectrum.npy --fsky 0.8
```

---

## CLI Command Reference

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
For the full reference of the analysis suite, please refer to the [r_analysis Documentation](r_analysis.md).
