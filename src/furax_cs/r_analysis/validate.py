import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from furax._instruments.sky import get_noise_sigma_from_instrument
from furax_cs.data.instruments import get_instrument
from furax.obs import negative_log_likelihood
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_healpy.clustering import get_cutout_from_mask
from furax.obs.landscapes import FrequencyLandscape

from .logging_utils import info, warning, error , success
from .utils import index_run_data
from .plotting import save_or_show , PLOT_OUTPUTS


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
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    @jax.jit
    def grad_nll(params):
        return jax.grad(negative_log_likelihood_fn)(
            params, nu=nu, N=N, d=noised_d, patch_indices=patch_indices,
        )

    @jax.jit
    def nll(params):
        return negative_log_likelihood_fn(
            params, nu=nu, N=N, d=noised_d, patch_indices=patch_indices,
        )

    # 3. Compute Validation Metrics
    steps = jnp.arange(-steps_range, steps_range + 1) # inclusive range
    results = {}

    print("Computing NLLs and Gradients for multiple scales...")

    for scale in scales:
        print(f"  Processing scale: {scale:.1e}")

        # Calculate perturbations for this scale
        # Shape: (n_steps, 1)
        perturbations = steps.reshape(-1, 1) * scale

        # Broadcast perturbations to parameter shape: (n_steps, n_patches)
        final_params_perturbed = jax.tree.map(
            lambda p: p.reshape(1, -1) + perturbations, final_params
        )

        # Vectorized computation
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


def plot_gradient_validation(validation_results, title_suffix=""):
    """
    Generate a 2x2 plot of NLL and Gradient Norms across scales.
    """
    scales = validation_results["scales"]
    steps = validation_results["steps"]
    results = validation_results["results"]

    # Setup Plotting Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Optimization Verification {title_suffix}", fontsize=16)

    # Colors for different scales
    # Use standard python list if jax array issues arise, but jnp usually fine
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(scales)))

    for i, scale in enumerate(scales):
        data = results[scale]
        
        label = f"Scale {scale:.1e}"
        color = colors[i]
        
        # Unpack data
        nlls = data["NLL"]
        g_bd = data["grads_beta_dust"]
        g_bp = data["grads_beta_pl"]
        g_td = data["grads_temp_dust"]

        # Plot 1: Negative Log Likelihood
        ax = axes[0, 0]
        ax.plot(steps, nlls, "o-", linewidth=2, color=color, label=label, alpha=0.8)

        # Plot 2: Gradient Norm - Beta Dust
        ax = axes[0, 1]
        ax.plot(steps, g_bd, "s-", linewidth=2, color=color, label=label, alpha=0.8)

        # Plot 3: Gradient Norm - Beta PL
        ax = axes[1, 0]
        ax.plot(steps, g_bp, "^-", linewidth=2, color=color, label=label, alpha=0.8)

        # Plot 4: Gradient Norm - Temp Dust
        ax = axes[1, 1]
        ax.plot(steps, g_td, "d-", linewidth=2, color=color, label=label, alpha=0.8)

    # Common Formatting
    plot_configs = [
        (axes[0, 0], "Negative Log-Likelihood", "NLL"),
        (axes[0, 1], "Gradient Norm: Beta Dust", "L2 Norm"),
        (axes[1, 0], "Gradient Norm: Beta PL", "L2 Norm"),
        (axes[1, 1], "Gradient Norm: Temp Dust", "L2 Norm"),
    ]

    for ax, title, ylabel in plot_configs:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Perturbation Steps (x Scale)")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Solution" if ax == axes[0,0] else "")
        if ax == axes[0, 0]: # Only legend on first plot to save space
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or Show
    # If running in CLI, usually better to save. We default to show for now unless configured.
    os.makedirs(PLOT_OUTPUTS, exist_ok=True)
    save_or_show(title_suffix , 'png')
    success(f"Validation plot saved to {title_suffix}.png")
    # plt.show() # Uncomment if interactive


def run_validate(matched_results, names , nside, instrument, steps, noise_ratio, scales):
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
                indices, = np.where(mask_arr)
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
                plot_title = f"{base_name}_seed_{run_idx}"
                
                plot_gradient_validation(val_res, title_suffix=plot_title)