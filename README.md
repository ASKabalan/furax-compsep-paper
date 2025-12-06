# FURAX Component Separation

This project compares CMB component separation frameworks (FURAX vs FGBuster) and implements adaptive clustering techniques.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd furax-compsep-paper
    ```

2.  **Download Large Files (Important!)**:
    This project uses Git LFS for large data files (masks). You must pull them manually if they weren't downloaded during clone:
    ```bash
    git lfs pull
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -e .
    ```
    *Note: Requires JAX. For GPU support, install `jax[cuda]` first.*

## Data Management

Before running analysis, you need to generate the frequency maps.

### 1. Generate Data (CLI)

Use the provided command-line tool to generate cached data:

```bash
# Generate frequency maps (simulated observations)
generate_data --nside 64 --sky c1d1s1
```

*   `--nside`: HEALPix resolution (e.g., 64, 128).
*   `--sky`: Sky model tag (e.g., `c1d1s1` for CMB + Dust + Synchrotron).

Note: Galactic masks are automatically generated and cached on first use.

### 2. Load Maps in Code

You can load the generated data directly in your Python scripts:

```python
from furax_cs.data.generate_maps import load_from_cache, get_mask
import healpy as hp

# 1. Load Frequency Maps
# Returns: frequencies (Hz), maps (shape: [freqs, 3, npix])
nu, freq_maps = load_from_cache(nside=64, sky="c1d1s1")

print(f"Loaded {len(nu)} frequencies.")
print(f"Map shape: {freq_maps.shape}")  # (n_freq, 3, n_pix)

# 2. Load a Galactic Mask
# Available: GAL020, GAL040, GAL060 (and combinations like GAL020+GAL040)
mask = get_mask("GAL020", nside=64)
```

## Example: Simple Adaptive Clustering

This example demonstrates how to perform spherical K-means clustering on the sky maps, similar to `notebooks/02_KMeans_Adaptive_Component_Separation.ipynb`.

```python
import jax
import jax.numpy as jnp
import healpy as hp
from jax_healpy.clustering import find_kmeans_clusters, normalize_by_first_occurrence
from furax_cs.data.generate_maps import get_mask

# Configuration
nside = 64
n_clusters = 10
mask_name = "GAL020"

# 1. Get Mask Indices
mask = get_mask(mask_name, nside=nside)
(indices,) = jnp.where(mask == 1)

# 2. Perform Spherical K-means Clustering
# We cluster the observed pixels into regions
print(f"Clustering {len(indices)} pixels into {n_clusters} regions...")

key = jax.random.PRNGKey(42)
clusters = find_kmeans_clusters(
    mask,
    indices,
    n_clusters,
    key,
    max_centroids=n_clusters,
    initial_sample_size=1
)

# Normalize cluster IDs for easier handling
clusters_normalized = normalize_by_first_occurrence(clusters, n_clusters, n_clusters)

# 3. Visualize
hp.mollview(clusters_normalized, title=f"Sky Clusters (N={n_clusters})")
```
