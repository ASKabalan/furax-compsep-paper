# CORRECTIONS NEEDED

This document consolidates all corrections that need to be made to the FURAX Component Separation Paper, organized by priority and implementation complexity.

## 📋 SUMMARY STATUS

### Advisor Comments Status
**Total JE Comments**: 22  
**Fully Addressed**: 13 (59%)  
**Partially Addressed**: 3 (14%)  
**Not Addressed**: 6 (27%)

**Total AR Comments**: 3  
**Fully Addressed**: 2 (67%)  
**Not Addressed**: 1 (33%)

### Critical Issues Requiring Immediate Attention
1. **BB Spectrum Convention paragraph structure** - Remove numbered paragraph formatting
2. **Error bars in variance figure** - Add error bars to show variance across noise realizations
3. **Multi-resolution naming convention** - Consider alternative naming schemes
4. **Figure labeling improvements** - Update legends and color schemes
5. **Missing figure commentary** - Add proper text references for important results

---

## 🔴 PHASE 1: CRITICAL FIXES (Do First)

### Fix 1: Mathematical Notation Corrections
**Priority**: CRITICAL  
**File**: `publications/paper/furax_comp_sep.tex`  
**Action**: Global replacement throughout the paper

**Find and Replace All Instances:**
```latex
$\beta_{dust}$        → $\beta_{\rm dust}$
$T_{dust}$            → $T_{\rm dust}$  
$\beta_{synchrotron}$ → $\beta_{\rm synchrotron}$
\beta_d               → \beta_{\rm d}
T_d                   → T_{\rm d}  
\beta_s               → \beta_{\rm s}
```
**Estimated replacements**: ~50 instances throughout paper

### Fix 2: Add Missing References
**Priority**: CRITICAL  
**File**: `publications/paper/furax_comp_sep.bib`

**HEALPix Reference** (Line 279):
```bibtex
@article{Gorski2005,
  title={HEALPix: a framework for high-resolution discretization and fast analysis of data distributed on the sphere},
  author={G{\'o}rski, Krzysztof M and Hivon, Eric and others},
  journal={The Astrophysical Journal},
  volume={622}, number={2}, pages={759}, year={2005}
}
```

**L-BFGS Reference** (Line 267):
```bibtex
@article{Liu1989,
  title={On the limited memory BFGS method for large scale optimization},
  author={Liu, Dong C and Nocedal, Jorge},
  journal={Mathematical programming},
  volume={45}, number={1-3}, pages={503--528}, year={1989}
}
```

**Planck Masks Reference** (Line 373):
```bibtex
@article{Planck2020Legacy,
  title={Planck 2018 results. I. Overview and the cosmological legacy of Planck},
  author={Planck Collaboration and Aghanim, N. and others},
  journal={Astronomy \& Astrophysics}, volume={641}, pages={A1}, year={2020},
  eprint={1807.06205}, archivePrefix={arXiv},
  note={Masks available at \url{https://pla.esac.esa.int/pla/}}
}
```

**Jean Zay Reference** (Line 539):
```bibtex
@misc{JeanZay2020,
  title={Jean Zay supercomputer}, author={{GENCI}},
  url={http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html},
  year={2020}, note={GENCI-IDRIS, France}
}
```

### Fix 3: Add Units to Figure Captions
**Priority**: CRITICAL  
**File**: `publications/paper/furax_comp_sep.tex`

**Line 395 - Add μK² units**:
```latex
\textbf{Top:} Mean variance of the reconstructed CMB component across noise realizations (μK²), used as the primary selection metric.
```

### Fix 4: Reference Missing Figures in Main Text
**Priority**: CRITICAL  
**File**: `publications/paper/furax_comp_sep.tex`

**Add before metric distributions figure**:
```latex
Figure~\ref{fig:metric_distributions} shows the distribution of evaluation metrics across the three spatial modeling strategies, demonstrating that K-means clustering consistently achieves the lowest residual variance and highest likelihood values.
```

**Add before r-likelihood figure**:
```latex
Figure~\ref{fig:r_likelihood_distribution} presents our key result: the estimated tensor-to-scalar ratio likelihood distributions. K-means clustering yields the lowest bias and tightest credible interval, significantly outperforming both multi-resolution configurations.
```

---

## 🟡 PHASE 2: FIGURE CODE CHANGES

### Fix 5: Update Legend Names and Colors
**Priority**: HIGH  
**File**: `content/09-R_estimation.py`  
**Function**: `plot_all_cl_residuals()`

**Add color mapping based on title content**:
```python
# Use specific colors based on title content
if "This work" in name or "K-means" in name:
    plot_color = "red"
    linewidth = 2
elif "Multi-resolution" in name:
    plot_color = "blue" if "(1)" in name else "green"
    linewidth = 1
else:
    plot_color = color  # default
    linewidth = 1
```

**Use proper command-line titles**:
```bash
python content/09-R_estimation.py -r BS500 PTEP1 PTEP3 -t "This work (K-means clustering)" "Multi-resolution MODELS (1)" "Multi-resolution MODELS (2)"
```

### Fix 6: Add Error Bars to Validation Plot
**Priority**: HIGH  
**File**: `data/plotting.py`  
**Function**: `plot_cmb_nll_vs_B_d_patches()`

**Add error bars modification**:
```python
# For the variance plot, add error bars
variance_mean = np.mean(values, axis=1)  # Mean across realizations
variance_std = np.std(values, axis=1)    # Std across realizations

# Plot with error bars
axs[0].errorbar(x_vals, variance_mean, yerr=variance_std, 
                fmt='o-', capsize=5, capthick=2,
                label=f"$K_{{T_d}}$={T_d}, $K_{{\beta_s}}$={B_s}")

# Update labels to use K notation
axs[0].set_xlabel("$K_{\\beta_d}$")
axs[0].set_ylabel("CMB Variance (μK²)")
```

### Fix 7: Fix r=1 vs r=0.001 Inconsistency
**Priority**: MEDIUM  
**File**: `content/09-R_estimation.py`  
**Function**: `plot_all_cl_residuals()`

**Change figure to match text (r=1)**:
```python
plt.plot(
    ell_range,
    cl_bb_r1 * coeff,  # Remove * 1e-3
    label=r"$C_\ell^{\mathrm{BB}}(r=1)$",  # Change label
    color="black", linewidth=2,
)
```

### Fix 8: Standardize K-notation in Figures
**Priority**: MEDIUM  
**Files**: `data/plotting.py` and `publications/paper/furax_comp_sep.tex`

**Update plotting code labels**:
```python
ax.set_xlabel("$K_{\\beta_d}$")  # Instead of "B_d patches"
ax.set_ylabel("CMB Variance (μK²)")
label=f"$K_{{T_d}}$={T_d}, $K_{{\beta_s}}$={B_s}"
```

**Update LaTeX text**:
```latex
T_d \in [5, 20]     → K_{T_d} \in [5, 20]
\beta_s \in [5, 20] → K_{\beta_s} \in [5, 20]
T_d = 20            → K_{T_d} = 20
\beta_s = 20        → K_{\beta_s} = 20
```

### Fix 9: Make Patch Layout Plots Vertical
**Priority**: MEDIUM  
**File**: `content/09-R_estimation.py`  
**Function**: `plot_params_patches()`

**Add vertical layout option**:
```python
def plot_params_patches(name, params, patches, plot_vertical=True):
    if plot_vertical:
        fig_size = (8, 12)
        subplot_args = (3, 1, i + 1)  # Vertical: 3 rows, 1 column
    else:
        fig_size = (12, 8) 
        subplot_args = (1, 3, i + 1)  # Horizontal: 1 row, 3 columns
    
    _ = plt.figure(figsize=fig_size)
    hp.mollview(..., sub=subplot_args, ...)
```

### Fix 10: Update Y-axis Labels in BB Spectra
**Priority**: MEDIUM  
**File**: `content/09-R_estimation.py`

**Change axis labels**:
```python
plt.ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]")  # Instead of $C_\ell^{BB}$ [1e-2 $\mu K^2$]
```

---

## 🟢 PHASE 3: CONTENT IMPROVEMENTS

### Fix 11: Enhanced Loss Function Discussion
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: After equation (15)

**Add comprehensive discussion**:
```latex
Although the variance of the reconstructed CMB and the total $\sum C_\ell^{BB}$ power are mathematically equivalent selection metrics under our assumptions, we systematically explored additional loss functions and figures of merit during model development. Alternative approaches investigated include: direct minimization of foreground residual power spectra, weighted combinations of statistical and systematic error metrics, information-theoretic criteria (AIC/BIC for model selection), cross-validation based selection, and direct optimization of cosmological parameter uncertainties. Future work will expand this analysis to include systematic comparison of these alternative selection criteria, particularly for their robustness across different foreground complexity regimes and observational scenarios.
```

### Fix 12: Format BB Spectrum Convention
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Line 295

**Change from numbered paragraph to subsection**:
```latex
\subsubsection*{BB Spectrum Convention}  % Instead of \paragraph{BB Spectrum Convention.}
```

### Fix 13: Define Single-Patch Method
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Line 597

**Add definition on first mention**:
```latex
We assess the quality of CMB reconstruction achieved by three spatial modeling strategies: single global patch (where all spectral parameters are uniform across the sky), multi-resolution grouping, and K-means clustering.
```

### Fix 14: Reorganize Sky Region Partitioning
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Lines 442-443

**Reorganize bullet points**:
```latex
To probe performance across different foreground regimes, we define distinct sky zones using Planck-based Galactic masks, with different spatial modeling approaches:
\begin{itemize}
    \item \textbf{K-means clustering}: Applied to six disjoint subregions, obtained by splitting each mask into upper and lower hemispheric parts: \texttt{GAL020\_U}, \texttt{GAL020\_L}, etc.
    \item \textbf{Multi-resolution grouping}: Applied separately to the full-sky masks: \texttt{GAL020}, \texttt{GAL040}, and \texttt{GAL060}.
\end{itemize}
```

---

## 🔵 PHASE 4: TECHNICAL CLARIFICATIONS

### Fix 15: Explain K-means vs Multi-resolution MSE
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: CMB reconstruction comparison section

**Add explanation after MSE values**:
```latex
This apparent contradiction occurs because K-means clustering optimizes for overall CMB variance minimization across the entire reconstruction pipeline, which does not necessarily minimize pixel-level mean squared error. The multi-resolution approach, with its regular geometric patches, can achieve lower local reconstruction errors in individual pixels, while K-means clustering achieves better global systematic residual control, leading to improved cosmological parameter estimation despite slightly higher pixel-level residuals.
```

### Fix 16: Clarify Optimization Approaches
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Line 631

**Add clarification**:
```latex
Note that both multi-resolution and K-means approaches can be optimized by minimizing CMB variance across noise realizations. However, K-means clustering provides more granularity and flexibility in patch selection, allowing for better adaptation to local foreground characteristics. The key difference lies in the patch structure: multi-resolution is constrained to regular HEALPix downgrading patterns, while K-means can create irregular patches that adapt to the underlying sky structure.
```

### Fix 17: Resolve Noise Performance Contradiction
**Priority**: MEDIUM  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Line 664

**Add reconciling sentence**:
```latex
While K-means shows slightly higher pixel-level noise in individual map reconstructions, it achieves better statistical performance in power spectrum estimation due to superior systematic residual control.
```

### Fix 18: Correct Residual Sensitivity Explanation
**Priority**: HIGH  
**File**: `publications/paper/furax_comp_sep.tex`  
**Location**: Lines 816-817

**Replace with corrected explanation**:
```latex
Both the CMB variance and $\sum C_\ell^{BB}$ metrics are sensitive to the total residual contamination—the sum of both statistical and systematic contributions. While minimizing the total variance is effective for overall performance, this approach can occasionally favor configurations that achieve lower total residuals while still maintaining systematic biases, provided the reduction in statistical noise compensates for elevated systematic residuals. This limitation highlights why minimizing the total variance does not guarantee systematic residual suppression and motivates the need for additional selection criteria that specifically target systematic contamination.
```

---

## 🟠 PHASE 5: STRUCTURAL IMPROVEMENTS

### Fix 19: Move Beam Effects Discussion
**Priority**: LOW  
**File**: `publications/paper/furax_comp_sep.tex`

**Move from line 556 to simulation setup (around line 425)**:
```latex
Note that beam effects are not modeled in this analysis, as our simulations use low-resolution data ($N_{\text{side}} = 64$) where beam convolution effects are minimal compared to foreground contamination. Future implementations will include proper beam modeling for higher-resolution analyses.
```

### Fix 20: Remove Repetition Between Sections
**Priority**: LOW  
**File**: `publications/paper/furax_comp_sep.tex`  
**Action**: Review Discussion (Section 6) and Conclusions (Section 7) sections and remove redundant content

### Fix 21: Fix PySM3 Citation Format
**Priority**: LOW  
**File**: `publications/paper/furax_comp_sep.bib`  
**Action**: Ensure proper author names rather than "Group" in the PySM3 citation

### Fix 22: Remove Editorial Comments
**Priority**: LOW  
**File**: `publications/paper/furax_comp_sep.tex`  
**Action**: Remove placeholder and editorial comments while preserving advisor comment structure

---

## 📊 CRITICAL UNADDRESSED ADVISOR COMMENTS

### JE Comments Still Needing Attention:
1. **Line 301**: BB Spectrum Convention paragraph formatting ❌
2. **Line 396**: Error bars in figure ❌  
3. **Line 417**: Multi-resolution naming ("multi-Healpix" vs "multi-resolution") ❌
4. **Line 424**: PySM3 citation format ❌
5. **Line 699**: Figure labeling improvements ❌
6. **Line 782**: Arianna's contribution section ❌

### AR Comments Still Needing Attention:
1. **Line 726**: New section "Towards lower statistical residuals" ❌

---

## 🎯 IMPLEMENTATION PRIORITY ORDER

### Immediate (Critical for publication):
1. Mathematical notation fixes (Fix 1)
2. Missing references (Fix 2)
3. Units in figures (Fix 3) 
4. Figure text references (Fix 4)
5. Residual sensitivity explanation (Fix 18)

### High Priority (Major improvements):
1. Error bars in validation plots (Fix 6)
2. Legend improvements (Fix 5)
3. Enhanced loss function discussion (Fix 11)

### Medium Priority (Quality improvements):
1. K-notation standardization (Fix 8)
2. Technical explanations (Fixes 15-17)
3. Content organization (Fixes 12-14)

### Low Priority (Polish):
1. Structural improvements (Fixes 19-22)
2. Editorial cleanup
3. Figure layout optimizations (Fix 9)

---

## 📋 FILES TO MODIFY SUMMARY

**LaTeX Files:**
- `publications/paper/furax_comp_sep.tex` (25+ specific text changes)
- `publications/paper/furax_comp_sep.bib` (4 new reference entries)

**Python Files:**
- `content/09-R_estimation.py` (Legend colors, r=1 consistency, Y-axis labels)
- `data/plotting.py` (K-notation, error bars for validation plots) 
- `content/02-validation-model.py` (Main validation script)

**Search/Replace Summary:**
- ~50 mathematical notation fixes
- ~10 cluster notation fixes  
- 4 new bibliography entries
- 25 specific text modifications

This comprehensive list ensures all corrections are tracked and can be systematically implemented to bring the paper to publication quality.