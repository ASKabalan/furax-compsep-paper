# Notebooks

This directory contains Jupyter notebooks for analyzing CMB component separation results and demonstrating the FURAX/FGBuster comparison framework.

## Overview

The notebooks are organized into two categories:
1. **Analysis notebooks** (01-04): Detailed analyses of different component separation methods
2. **Plotting notebooks** (80-series): Result visualization and figure generation

## Notebook Descriptions

### 01_FGBuster_vs_FURAX_Comparison.ipynb

**Purpose**: Framework validation and performance comparison

This notebook demonstrates the fundamental comparison between the traditional FGBuster (NumPy-based) and modern FURAX (JAX-based) frameworks for CMB component separation.

**Contents**:
- Side-by-side implementation comparison
- Validation that both frameworks produce consistent results
- Performance benchmarking (CPU vs GPU)
- Scaling analysis across different HEALPix resolutions
- Solver comparisons (TNC, L-BFGS)

**Key Outputs**:
- Validation plots showing agreement between frameworks
- Timing comparisons and weak scaling curves
- Memory usage analysis

---

### 02_KMeans_Adaptive_Component_Separation.ipynb

**Purpose**: K-means adaptive component separation method analysis

Explores the K-means clustering approach for spatially-varying spectral parameters, where the sky is divided into patches with distinct foreground properties.

**Contents**:
- K-means clustering methodology
- Adaptive patch determination
- Impact of cluster count on CMB reconstruction quality
- Comparison with uniform patch models
- Parameter recovery analysis (beta_d, temp_d, beta_s)

**Key Outputs**:
- Patch assignment visualizations
- CMB reconstruction quality vs. cluster count
- Spectral parameter maps
- Variance analysis

**Key Parameters**:
- `-pc` (patch configuration): Controls the number of clusters for each parameter

---

### 03_PTEP_Multi_Resolution_Component_Separation.ipynb

**Purpose**: Multi-resolution (PTEP) component separation analysis

Investigates hierarchical multi-resolution approaches where different sky regions are analyzed at different HEALPix resolutions based on galactic emission complexity.

**Contents**:
- Multi-resolution methodology (PTEP approach)
- Resolution selection strategies
- Galactic mask integration (GAL020, GAL040, GAL060)
- Computational efficiency gains
- Reconstruction accuracy vs. resolution trade-offs

**Key Outputs**:
- Multi-resolution sky maps
- Resolution assignment visualizations
- Computational cost comparisons
- CMB reconstruction residuals

**Key Parameters**:
- `-ud` (updown parameters): Specifies resolution ladder (e.g., `64 32 16`)

---

### 04_Scripts_and_Analysis_Workflow.ipynb

**Purpose**: Complete end-to-end workflow demonstration

A comprehensive tutorial notebook that demonstrates the full analysis pipeline from data generation to final r-statistic estimation.

**Contents**:
- Data generation and caching workflow
- Running component separation scripts (`kmeans-model`, `ptep-model`)
- Using the `r_analysis` tool (note: subcommands like `snap` or `plot` must come first)
- Interpreting results
- Generating publication-quality figures
- Best practices and troubleshooting

**Key Sections**:
1. Environment setup
2. Data preparation
3. Running experiments (local and HPC)
4. Result organization
5. Analysis pipeline execution
6. Visualization and interpretation
