# BLUEPRINT.md

This document provides complete documentation for figure generation, experiment workflows, and advisor comment protocols for the FURAX Component Separation Paper.

## 🔴 CRITICAL REQUIREMENT - Advisor Comment Preservation Protocol

**IMPORTANT**: When implementing fixes in the LaTeX paper, DO NOT delete or override the original advisor comments marked with `\je{}` and `\ar{}`. Instead:

1. **Keep all original advisor comments intact** in the LaTeX file
2. **Add colored `\wk{}` markup** showing how each comment was addressed
3. **Use the format**: `\je{original comment} \wk{Fixed: [specific action taken]}`

**Example Implementation:**
```latex
% Original advisor comment with response
\je{put $\rm{dust}$ instead of $\beta_{dust}$} \wk{Fixed: Updated all spectral parameter notation throughout paper to use $\beta_{\rm dust}$, $T_{\rm dust}$, and $\beta_{\rm synchrotron}$ format}

% The actual fix in the text
with $\boldsymbol{\beta}$ denoting the set of corresponding spectral parameters ($\beta_{\rm dust}$, $T_{\rm dust}$, $\beta_{\rm synchrotron}$).
```

### Comment Handling Protocol

When implementing fixes, follow this protocol:

**DO NOT:**
- Delete original `\je{}` or `\ar{}` advisor comments
- Override advisor markup with your own changes
- Remove the colored highlighting from advisor feedback

**DO:**
- Keep all original advisor comments visible in the LaTeX source
- Add `\wk{Response text}` after each addressed comment
- Use green coloring for your responses: `\newcommand{\wk}[1]{\textcolor{green}{#1}}`
- Document the specific action taken for each comment

This maintains transparency of the review process while documenting all implemented changes.

---

## 📊 Complete Figure Generation Reference

This section documents how each figure in the paper is generated, including source files, functions, and exact commands needed.

### Notebooks and Illustration Figures

| Figure | Source | Function/Section | Notes |
|--------|--------|------------------|-------|
| `figures/kmeans_clustering.pdf` | `content/80-illustraions.ipynb` | - | Spherical K-means example |
| `figures/runtime_comparison.pdf` | `content/80-plot_runs.ipynb` | - | Performance comparison |
| `figures/multi_resolution_pixel_grouping.pdf` | `content/80-illustraions.ipynb` | - | Multi-resolution structure |

### Validation Figures

| Figure | Source Script | Command | Output Location |
|--------|---------------|---------|-----------------|
| `figures/validation_likelihood_vs_variance.pdf` ⭐ | `content/03-noise-model.py` | `python 03-noise-model.py -n 64 -ns 20 -nr 0.2 -m GAL020 -p` | `results/noise_validation_modelGAL020_20/validation_likelihood_vs_variance.pdf` |

**Note**: ⭐ marks IMPORTANT figures. The validation figure is generated in results folder and must be renamed/copied to figures folder.

### Main Results Figures (via `09-R_estimation.py`)

**Primary Command:**
```bash
python content/09-R_estimation.py -r BS500 PTEP1 PTEP3 -t "This work (K-means clustering)" "Multi-resolution MODELS (1)" "Multi-resolution MODELS (2)" -a
```

**Output Location:** All PDFs generated in `content/plots/` folder

| Figure | Function | Description | Layout Notes |
|--------|----------|-------------|--------------|
| `figures/kmeans_patch_layout.pdf` | `plot_params_patches()` | K-means patch structure | Current: horizontal, Paper: vertical |
| `figures/multires_patch_layout.pdf` | `plot_params_patches()` | Multi-resolution patches | Current: horizontal, Paper: vertical |
| `figures/cmb_recon.pdf` | `plot_cmb_recon()` + `plot_all_cmb()` | CMB reconstruction comparison | - |
| `figures/bb_residual_spectra.pdf` | `plot_all_cl_residuals()` | B-mode power spectra | Needs legend fixes |
| `figures/variance_likelihood_distributions.pdf` | `plot_all_variances()` | Metric distributions | Needs error bars |
| `figures/r_likelihood_distribution.pdf` | `plot_all_r_estimation()` | r-parameter estimation | Missing text reference |
| `figures/r_vs_clusters.pdf` | `plot_r_vs_clusters()` | r vs cluster count | Missing text reference |

### Parameter Distribution Figures

**Files to copy**:
- `content/plots/params_This work (K-means clustering).pdf` → `publications/paper/figures/params_kmeans.pdf`
- `content/plots/params_Multi-resolution MODELS (1).pdf` → `publications/paper/figures/params_multires.pdf`

---

## 🔄 Complete Figure Generation Workflow

### Step 1: Run Validation Script
```bash
cd content
python 03-noise-model.py -n 64 -ns 20 -nr 0.2 -m GAL020 -p
```
**Output**: `results/noise_validation_modelGAL020_20/validation_likelihood_vs_variance.pdf`

### Step 2: Run Main Plotting Script
```bash
cd content  
python 09-R_estimation.py -r BS500 PTEP1 PTEP3 -t "This work (K-means clustering)" "Multi-resolution MODELS (1)" "Multi-resolution MODELS (2)" -a
```
**Output**: All main result PDFs in `content/plots/`

### Step 3: Copy/Rename Files
Copy files from output locations to `publications/paper/figures/`:

```bash
# Copy validation plot
cp results/noise_validation_modelGAL020_20/validation_likelihood_vs_variance.pdf publications/paper/figures/

# Copy main result plots  
cp content/plots/*.pdf publications/paper/figures/

# Rename parameter files
mv "content/plots/params_This work (K-means clustering).pdf" publications/paper/figures/params_kmeans.pdf
mv "content/plots/params_Multi-resolution MODELS (1).pdf" publications/paper/figures/params_multires.pdf
```

### Step 4: Check Layout
Some functions generate horizontal layouts but paper uses vertical - check and regenerate if needed.

---

## 🚀 Experiment Execution Commands

### Individual Scripts (run from content/ directory)

#### Benchmarking Scripts
```bash
# Benchmark basic component separation
python 01-bench_bcp.py -n 64 -s -l
# -n: HEALPix nside resolution, -s: show stats, -l: enable logging

# Benchmark clustering performance  
python 01-bench_clusters.py -n 64 -cl 100
# -cl: number of clusters to test
```

#### Model Validation
```bash
# Run validation model
python 02-validation-model.py -n 64 -m GAL020
# -n: nside, -m: mask type (GAL020/040/060, GALACTIC)

# Noise model testing
python 03-noise-model.py -n 64 -ns 100 -nr 1.0 -m GAL020
# -ns: number of samples, -nr: noise ratio (0.0-1.0)
```

#### Main K-means Clustering Method (Key Innovation)
```bash
python 08-KMeans-model.py -n 64 -pc 100 5 1 -tag c1d1s1 -m GAL020 -i LiteBIRD
# -n: nside, -pc: clusters for [dust_temp, dust_beta, sync_beta]
# -tag: simulation config, -m: mask, -i: instrument
```

#### Tensor-to-scalar Ratio Estimation
```bash
python 09-R_estimation.py  # Estimate r parameter
```

#### Other Methods for Comparison
```bash
# Multi-resolution
python 05-PTEP-model.py -n 64 -ud 64 0 2 -tag c1d1s1 -m GAL020 -i LiteBIRD

# Single patch
python 07-single_patch.py -n 64 -tag c1d1s1 -m GAL020 -i LiteBIRD
```

### Full Experiment Suite (SLURM)
```bash
cd content
./99-run_all_jobs.sh    # Submit all SLURM jobs for complete analysis
```

---

## 📋 Title Mapping for Command Line

When running the main plotting script, use these title mappings:

- `BS500` results → `"This work (K-means clustering)"`
- `PTEP1` results → `"Multi-resolution MODELS (1)"`  
- `PTEP3` results → `"Multi-resolution MODELS (2)"`

**Example command with proper titles:**
```bash
python content/09-R_estimation.py -r BS500 PTEP1 PTEP3 -t "This work (K-means clustering)" "Multi-resolution MODELS (1)" "Multi-resolution MODELS (2)" -a
```

---

## 🛠️ Data Management and Troubleshooting

### Cache Management
```bash
# View cached frequency maps
ls content/freq_maps_cache/

# Clear cache if regeneration needed
rm content/freq_maps_cache/freq_maps_nside_*.pkl
```

**Cache Details**:
- **Location**: Frequency maps cached in `content/freq_maps_cache/` to avoid expensive PySM3 regeneration
- **Naming pattern**: `freq_maps_nside_{nside}_{noise_setting}_{config_tag}.pkl`
- **Cache keys**: nside resolution, noise vs no_noise, configuration tag (c1d1s1, etc.)
- **Storage**: Cache files can be large (~GB) - monitor disk usage on clusters

### GPU and JAX Troubleshooting
```bash
# Check GPU availability
python -c "import jax; print('JAX devices:', jax.devices())"

# Check GPU status (SLURM environments)
nvidia-smi

# Enable JAX debugging
export EQX_ON_ERROR=nan
export JAX_TRACEBACK_FILTERING=off
```

### Results and Plots
```bash
# View experiment results
ls results/

# View generated plots  
ls content/plots/
```

**Results use naming**: `{method}_{config}_{instrument}_{mask}_{samples}/`

---

## 🔧 Key Python Functions to Modify

### Functions in `content/09-R_estimation.py`:
1. **`plot_all_cl_residuals()`**: Legend color coding based on titles
2. **`plot_all_r_estimation()`**: Consistent naming and colors  
3. **`plot_params_patches()`**: Make vertical layout for paper

### Functions in `data/plotting.py`:
1. **`plot_cmb_nll_vs_B_d_patches()`**: Add error bars and K-notation
2. **Global axis labeling**: Update K-notation throughout validation figures

### Main validation script:
- **`content/02-validation-model.py`**: Main validation script (calls plotting functions)

---

## 📈 Performance Notes

- **Full study**: 3200 clustering configurations × 100 noise realizations × 6 sky regions = ~1.92 million component separation runs
- **Completion time**: Under 30 hours on 32 A100 GPUs (Jean Zay supercomputer)  
- **Performance gain**: >20× speedup over CPU-based frameworks like FGBuster
- **Scalability**: Distributed execution with fault tolerance and checkpointing

---

This blueprint provides all necessary information for reproducing figures, running experiments, and maintaining the collaborative review process with advisor comments.