import os
from functools import partial

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
from furax import Config
from furax._instruments.sky import get_noise_sigma_from_instrument
from furax.obs import negative_log_likelihood
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_healpy.clustering import get_cutout_from_mask

from ..logging_utils import error, info, success
from .plotting import PLOT_OUTPUTS, save_or_show
from .utils import index_run_data


def compute_gradient_validation(
    # Data & Parameters
    final_params,
    masked_d,
    patch_indices,
    mask_indices,
    # Instrument & Config
    instrument,
    nside,
    # Validation Settings
    scales,
    steps_range,
    noise_ratio=0.0,
    run_idx=0,
):
    """
    Compute NLL and gradient norms for perturbed parameters across multiple scales.

    Parameters
    ----------
    final_params : dict
        The optimized parameters (beta_dust, beta_pl, temp_dust).
    masked_d : Stokes
        The input data (masked).
    patch_indices : dict
        Patch assignments.
    mask_indices : jnp.ndarray
        Indices where mask is 1.
    instrument : Instrument
        Instrument object.
    nside : int
        Healpix nside.
    scales : list of float
        Scales of perturbation to validate.
    steps_range : int
        Number of steps +/- to perturb.
    noise_ratio : float
        Ratio of noise to add (0.0 for noiseless validation).
    run_idx : int
        Run index (used for RNG seeding).
    """
    info(f"is X64 enabled: {jax.config.jax_enable_x64}")
    # 1. Construct Noise Operator & Data
    key = jax.random.PRNGKey(run_idx)
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    # Generate noise based on instrument specs
    white_noise = f_landscapes.normal(key) * noise_ratio
    white_noise = get_cutout_from_mask(white_noise, mask_indices, axis=1)

    sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
    noise = white_noise * sigma
    noised_d = masked_d + noise

    # Setup Diagonal Noise Operator (N)
    # If noise_ratio is 0, we use identity-like weight (1.0) to avoid singular N
    small_n = (sigma * noise_ratio) ** 2 if noise_ratio > 0 else 1.0
    N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

    # 2. Define Likelihood Functions
    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
    nu = instrument.frequency

    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    @jax.jit
    def grad_nll(params):
        return jax.grad(negative_log_likelihood_fn)(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    @jax.jit
    def nll(params):
        return negative_log_likelihood_fn(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    # 3. Compute Validation Metrics
    steps = jnp.arange(-steps_range, steps_range + 1)  # inclusive range
    results = {}

    info("Computing NLLs and Gradients for multiple scales...")

    for scale in scales:
        info(f"  Processing scale: {scale:.1e}")

        # Calculate perturbations for this scale
        # Shape: (n_steps, 1)
        perturbations = steps.reshape(-1, 1) * scale

        # Broadcast perturbations to parameter shape: (n_steps, n_patches)
        final_params_perturbed = jax.tree.map(
            lambda p: p.reshape(1, -1) + perturbations, final_params
        )

        # Vectorized computation
        with Config(solver=lx.CG(rtol=1e-6, atol=1e-10, max_steps=10000)):
            nlls = jax.vmap(nll)(final_params_perturbed)
            grads = jax.vmap(grad_nll)(final_params_perturbed)

        # Compute Norms of gradients
        grads_beta_dust_norm = jnp.linalg.norm(grads["beta_dust"], axis=1)
        grads_beta_pl_norm = jnp.linalg.norm(grads["beta_pl"], axis=1)
        grads_temp_dust_norm = jnp.linalg.norm(grads["temp_dust"], axis=1)

        results[scale] = {
            "NLL": nlls,
            "grads_beta_dust": grads_beta_dust_norm,
            "grads_beta_pl": grads_beta_pl_norm,
            "grads_temp_dust": grads_temp_dust_norm,
        }

    return {"scales": scales, "steps": steps, "results": results}


def plot_gradient_validation(validation_results, file_name, title, subfolder=None):
    """
    Generate a 2x2 plot of NLL and Gradient Norms across scales.
    """
    scales = validation_results["scales"]
    steps = validation_results["steps"]
    results = validation_results["results"]

    # Setup Plotting Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Optimization Verification {title}", fontsize=16)

    # Colors for different scales
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(scales)))

    for i, scale in enumerate(scales):
        data = results[scale]
        color = colors[i]

        # Unpack data
        nlls = data["NLL"]
        g_bd = data["grads_beta_dust"]
        g_bp = data["grads_beta_pl"]
        g_td = data["grads_temp_dust"]

        # --- Calculate Stats for Legend ---
        # 1. Identify value at Step 0 (The Solver's Solution)
        idx_zero = jnp.argmin(jnp.abs(steps))
        nll_zero = nlls[idx_zero]

        # 2. Identify global Minimum in this scan
        nll_min = jnp.min(nlls)

        # 3. Calculate Differences
        abs_diff = nll_zero - nll_min
        rel_diff = abs(abs_diff / nll_min) if nll_min != 0 else 0.0

        # Create specific label for NLL plot containing the stats
        # Using newlines to keep the legend box compact width-wise
        nll_label = (
            f"Scale {scale:.1e}\n"
            f" Sol:  {nll_zero:.7e}\n"
            f" Min:  {nll_min:.7e}\n"
            f" Diff: {abs_diff:.2e}\n"
            f" Rel:  {rel_diff:.2e}"
        )
        base_label = f"Scale {scale:.1e}"
        # ----------------------------------

        # Plot 1: Negative Log Likelihood (Uses detailed legend)
        ax = axes[0, 0]
        # We plot nlls directly, but the legend explains the gap
        ax.plot(steps, nlls, "o-", linewidth=2, color=color, label=nll_label, alpha=0.8)

        # Plot 2: Gradient Norm - Beta Dust (Uses simple legend)
        ax = axes[0, 1]
        ax.plot(steps, g_bd, "s-", linewidth=2, color=color, label=base_label, alpha=0.8)

        # Plot 3: Gradient Norm - Beta PL
        ax = axes[1, 0]
        ax.plot(steps, g_bp, "^-", linewidth=2, color=color, label=base_label, alpha=0.8)

        # Plot 4: Gradient Norm - Temp Dust
        ax = axes[1, 1]
        ax.plot(steps, g_td, "d-", linewidth=2, color=color, label=base_label, alpha=0.8)

    # Common Formatting
    plot_configs = [
        (axes[0, 0], "Negative Log-Likelihood", "NLL"),
        (axes[0, 1], "Gradient Norm: Beta Dust", "L2 Norm"),
        (axes[1, 0], "Gradient Norm: Beta PL", "L2 Norm"),
        (axes[1, 1], "Gradient Norm: Temp Dust", "L2 Norm"),
    ]

    for ax, ax_title, ylabel in plot_configs:
        ax.set_title(ax_title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Perturbation Steps (x Scale)")
        ax.grid(True, alpha=0.3)

        # Vertical line at 0
        ax.axvline(
            0,
            color="red",
            linestyle="--",
            alpha=0.5,
            # We don't need a label here anymore since it's in the main legend
        )

        # Add legend
        # For NLL plot, move legend outside if it's too big, or keep inside best location
        if ax == axes[0, 0]:
            ax.legend(fontsize="small", loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    base_dir = os.path.join(PLOT_OUTPUTS, subfolder) if subfolder else PLOT_OUTPUTS
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, "png", subfolder=subfolder)
    success(f"Validation plot saved to {file_name}.png")


def run_validate(matched_results, names, nside, instrument, steps, noise_ratio, scales):
    """Entry point for 'validate' subcommand."""

    for name, (kw, matched_folders) in zip(names, matched_results.items()):
        folders, run_indices = matched_folders

        # Normalize run_indices
        if isinstance(run_indices, int):
            run_indices = [run_indices]
        elif isinstance(run_indices, tuple) and len(run_indices) == 2:
            run_indices = list(range(run_indices[0], run_indices[1] + 1))

        for folder in folders:
            results_path = f"{folder}/results.npz"
            best_params_path = f"{folder}/best_params.npz"
            mask_path = f"{folder}/mask.npy"

            try:
                full_results = dict(np.load(results_path))
                best_params = dict(np.load(best_params_path))
                mask_arr = np.load(mask_path)
                (indices,) = np.where(mask_arr)
            except (FileNotFoundError, OSError) as e:
                error(f"Failed to load data for {folder}: {e}")
                continue

            d_stokes = Stokes.from_stokes(Q=best_params["I_D"][:, 0], U=best_params["I_D"][:, 1])

            for run_idx in run_indices:
                info(f"Validating run index {run_idx} in folder '{folder}'")

                # 1. Prepare Data
                run_data_sliced = index_run_data(full_results, run_idx)

                # Extract the specific parameters that minimized variance/NLL for this run
                # (Assuming 'value' is stored, otherwise use NLL or just index -1)
                nll = run_data_sliced["NLL"]
                indx = np.argmin(nll)

                patches = {
                    "beta_dust_patches": run_data_sliced["beta_dust_patches"],
                    "beta_pl_patches": run_data_sliced["beta_pl_patches"],
                    "temp_dust_patches": run_data_sliced["temp_dust_patches"],
                }

                # Handle if history is stored with shape (n_iter, n_patches)
                final_params = {
                    "beta_dust": run_data_sliced["beta_dust"][indx],
                    "beta_pl": run_data_sliced["beta_pl"][indx],
                    "temp_dust": run_data_sliced["temp_dust"][indx],
                }

                # 2. Compute Validation Stats
                val_res = compute_gradient_validation(
                    final_params=final_params,
                    masked_d=d_stokes,
                    patch_indices=patches,
                    mask_indices=indices,
                    instrument=instrument,
                    nside=nside,
                    scales=scales,
                    steps_range=steps,
                    noise_ratio=noise_ratio,
                    run_idx=run_idx,
                )

                # 3. Plot Results
                base_name = os.path.basename(folder.rstrip("/"))
                file_name = f"{base_name}_seed_{run_idx}"

                plot_gradient_validation(val_res, title=name, file_name=file_name, subfolder=kw)
