import os
from functools import partial
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
from furax import Config
from furax._instruments.sky import FGBusterInstrument
from furax.obs import negative_log_likelihood
from furax.obs.stokes import Stokes
from furax_cs import generate_noise_operator
from jax_healpy.clustering import get_fullmap_from_cutout
from jaxtyping import Array, Float, PyTree

from ..logging_utils import error, info, success
from .plotting import PLOT_OUTPUTS, save_or_show
from .utils import index_run_data


def parse_perturb_spec(spec: str, n_params: int) -> Array | None:
    """Parse perturbation spec: 'all', '-1', '0,1,2,3', or '0:30'.

    Returns None if spec is '-1' (skip this param).
    """
    if spec == "-1":
        return None
    if spec == "all":
        return np.arange(n_params)
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else n_params
        return np.arange(start, min(end, n_params))
    return np.array([int(x) for x in spec.split(",")])


def compute_gradient_validation(
    # Data & Parameters
    final_params: PyTree[Float[Array, " P"]],
    masked_d: Stokes,
    patch_indices: PyTree[Float[Array, " P"]],
    mask_indices: Array,
    # Instrument & Config
    instrument: FGBusterInstrument,
    nside: int,
    # Validation Settings
    scales: list[float],
    steps_range: int,
    noise_ratio: float = 0.0,
    run_idx: int = 0,
    # Perturbation specs per parameter
    perturb_beta_dust: str = "all",
    perturb_beta_pl: str = "all",
    perturb_temp_dust: str = "all",
    # Execution mode
    use_vmap: bool = True,
) -> dict[str, Any]:
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
    mask_indices : Array
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
    perturb_beta_dust : str
        Perturbation spec for beta_dust: 'all', '-1', '0:30', or '0,1,2'.
    perturb_beta_pl : str
        Perturbation spec for beta_pl.
    perturb_temp_dust : str
        Perturbation spec for temp_dust.
    use_vmap : bool
        If True (default), use vmap for vectorized computation.
        If False, use for-loop (slower but less memory).
    """
    info(f"is X64 enabled: {jax.config.jax_enable_x64}")
    # 1. Construct Noise Operator & Data
    key = jax.random.PRNGKey(run_idx)

    noised_d, N = generate_noise_operator(
        key, noise_ratio, mask_indices, nside, masked_d, instrument, stokes_type="QU"
    )

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
    def grad_nll(params: dict[str, Array]) -> dict[str, Array]:
        return jax.grad(negative_log_likelihood_fn)(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    @jax.jit
    def nll(params: dict[str, Array]) -> Array:
        return negative_log_likelihood_fn(
            params,
            nu=nu,
            N=N,
            d=noised_d,
            patch_indices=patch_indices,
        )

    # 3. Parse perturbation specs and create masks
    n_bd = final_params["beta_dust"].shape[0]
    n_bp = final_params["beta_pl"].shape[0]
    n_td = final_params["temp_dust"].shape[0]

    idx_bd = parse_perturb_spec(perturb_beta_dust, n_bd)
    idx_bp = parse_perturb_spec(perturb_beta_pl, n_bp)
    idx_td = parse_perturb_spec(perturb_temp_dust, n_td)

    # Create masks (1 where perturb, 0 elsewhere)
    mask_bd = np.zeros(n_bd)
    mask_bp = np.zeros(n_bp)
    mask_td = np.zeros(n_td)
    if idx_bd is not None:
        mask_bd[idx_bd] = 1
    if idx_bp is not None:
        mask_bp[idx_bp] = 1
    if idx_td is not None:
        mask_td[idx_td] = 1

    masks = {
        "beta_dust": jnp.array(mask_bd),
        "beta_pl": jnp.array(mask_bp),
        "temp_dust": jnp.array(mask_td),
    }

    # 4. Compute Validation Metrics
    steps = jnp.arange(-steps_range, steps_range + 1)  # inclusive range
    results = {}

    info("Computing NLLs and Gradients for multiple scales...")

    for scale in scales:
        info(f"  Processing scale: {scale:.1e}")

        # Calculate perturbations for this scale
        # Shape: (n_steps, 1)
        perturbations = steps.reshape(-1, 1) * scale

        # Apply perturbation with masks
        final_params_perturbed = {
            "beta_dust": final_params["beta_dust"].reshape(1, -1)
            + perturbations * masks["beta_dust"],
            "beta_pl": final_params["beta_pl"].reshape(1, -1) + perturbations * masks["beta_pl"],
            "temp_dust": final_params["temp_dust"].reshape(1, -1)
            + perturbations * masks["temp_dust"],
        }

        # Compute NLLs and gradients
        with Config(solver=lx.CG(rtol=1e-6, atol=1e-10, max_steps=10000)):
            if use_vmap:
                # Vectorized computation (faster but more memory)
                nlls = jax.vmap(nll)(final_params_perturbed)
                grads = jax.vmap(grad_nll)(final_params_perturbed)
            else:
                # For-loop approach (slower but less memory)
                n_steps = len(steps)
                nlls_list = []
                grads_list = []
                for i in range(n_steps):
                    params_i = jax.tree.map(lambda x: x[i], final_params_perturbed)
                    nlls_list.append(nll(params_i))
                    grads_list.append(grad_nll(params_i))
                nlls = jnp.stack(nlls_list)
                grads = jax.tree.map(lambda *xs: jnp.stack(xs), *grads_list)

        # Compute Norms of gradients
        grads_beta_dust_norm = jnp.linalg.norm(grads["beta_dust"], axis=1)
        grads_beta_pl_norm = jnp.linalg.norm(grads["beta_pl"], axis=1)
        grads_temp_dust_norm = jnp.linalg.norm(grads["temp_dust"], axis=1)

        results[scale] = {
            "NLL": nlls,
            "grads_beta_dust": grads_beta_dust_norm,
            "grads_beta_pl": grads_beta_pl_norm,
            "grads_temp_dust": grads_temp_dust_norm,
            "grads_raw": grads,  # Store raw gradients for grad-maps
        }

    return {"scales": scales, "steps": steps, "results": results}


def _get_markers(steps):
    """Get marker styles based on number of steps."""
    use_markers = len(steps) <= 10
    return {
        "nll": "o-" if use_markers else "-",
        "bd": "s-" if use_markers else "-",
        "bp": "^-" if use_markers else "-",
        "td": "d-" if use_markers else "-",
    }


def _plot_nll(
    validation_results: dict[str, Any] | list[dict[str, Any]],
    file_name: str,
    title: str,
    labels: list[str] | None = None,
    subfolder: str | None = None,
) -> None:
    """Plot NLL only."""
    # Handle single result or list of results
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
        labels = labels or [""]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"NLL Validation {title}", fontsize=16)

    # Colors for scales, linestyles for runs
    first_res = validation_results[0]
    scales = first_res["scales"]
    steps = first_res["steps"]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(scales)))
    linestyles = ["-", "--", ":", "-."]
    markers = _get_markers(steps)

    for run_idx, (val_res, label) in enumerate(zip(validation_results, labels)):
        ls = linestyles[run_idx % len(linestyles)]
        for scale_idx, scale in enumerate(scales):
            data = val_res["results"][scale]
            nlls = data["NLL"]
            color = colors[scale_idx]
            run_label = f"{label} {scale:.1e}" if label else f"Scale {scale:.1e}"
            ax.plot(
                steps,
                nlls,
                markers["nll"],
                linestyle=ls,
                linewidth=2,
                color=color,
                label=run_label,
                alpha=0.8,
            )

    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Perturbation Steps (x Scale)")
    ax.set_ylabel("NLL")
    ax.set_title("Negative Log-Likelihood")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="best")
    plt.tight_layout()

    base_dir = os.path.join(PLOT_OUTPUTS, subfolder) if subfolder else PLOT_OUTPUTS
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, "png", subfolder=subfolder)
    success(f"NLL plot saved to {file_name}.png")


def _plot_grad_norms(
    validation_results: dict[str, Any] | list[dict[str, Any]],
    file_name: str,
    title: str,
    labels: list[str] | None = None,
    subfolder: str | None = None,
) -> None:
    """Plot gradient norms (1x3)."""
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
        labels = labels or [""]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Gradient Norms {title}", fontsize=16)

    first_res = validation_results[0]
    scales = first_res["scales"]
    steps = first_res["steps"]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(scales)))
    linestyles = ["-", "--", ":", "-."]
    mkrs = _get_markers(steps)

    param_configs = [
        ("grads_beta_dust", "Beta Dust", mkrs["bd"]),
        ("grads_beta_pl", "Beta PL", mkrs["bp"]),
        ("grads_temp_dust", "Temp Dust", mkrs["td"]),
    ]

    for ax, (key, param_title, marker) in zip(axes, param_configs):
        for run_idx, (val_res, label) in enumerate(zip(validation_results, labels)):
            ls = linestyles[run_idx % len(linestyles)]
            for scale_idx, scale in enumerate(scales):
                data = val_res["results"][scale]
                grad_norm = data[key]
                color = colors[scale_idx]
                run_label = f"{label} {scale:.1e}" if label else f"Scale {scale:.1e}"
                ax.plot(
                    steps,
                    grad_norm,
                    marker,
                    linestyle=ls,
                    linewidth=2,
                    color=color,
                    label=run_label,
                    alpha=0.8,
                )

        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Perturbation Steps (x Scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title(f"Gradient: {param_title}")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize="small", loc="best")
    plt.tight_layout()

    base_dir = os.path.join(PLOT_OUTPUTS, subfolder) if subfolder else PLOT_OUTPUTS
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, "png", subfolder=subfolder)
    success(f"Gradient norms plot saved to {file_name}.png")


def _plot_grad_maps(
    validation_results: dict[str, Any],
    step_idx: int,
    patches: dict[str, Array],
    mask_indices: Array,
    nside: int,
    file_name: str,
    title: str,
    subfolder: str | None = None,
) -> None:
    """Plot gradient healpix maps at a specific step index."""
    scales = validation_results["scales"]
    steps = validation_results["steps"]
    results = validation_results["results"]

    # Convert step_idx (can be negative like -3) to array index
    step_array_idx = int(jnp.argmin(jnp.abs(steps - step_idx)))
    actual_step = steps[step_array_idx]

    # Use first scale
    scale = scales[0]
    grads_raw = results[scale]["grads_raw"]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Gradient Maps {title} (step={actual_step}, scale={scale:.1e})", fontsize=14)

    param_configs = [
        ("beta_dust", "beta_dust_patches", "Grad: Beta Dust"),
        ("beta_pl", "beta_pl_patches", "Grad: Beta PL"),
        ("temp_dust", "temp_dust_patches", "Grad: Temp Dust"),
    ]

    for i, (param_key, patch_key, param_title) in enumerate(param_configs):
        grad_values = grads_raw[param_key][step_array_idx]
        patch_idx = patches[patch_key]
        grad_at_pixels = grad_values[patch_idx]
        full_map = get_fullmap_from_cutout(grad_at_pixels, mask_indices, nside)

        hp.mollview(
            full_map,
            title=param_title,
            sub=(1, 3, i + 1),
            cmap="RdBu_r",
            bgcolor=(0.0,) * 4,
        )

    plt.tight_layout()

    base_dir = os.path.join(PLOT_OUTPUTS, subfolder) if subfolder else PLOT_OUTPUTS
    os.makedirs(base_dir, exist_ok=True)
    save_or_show(file_name, "png", subfolder=subfolder)
    success(f"Gradient maps saved to {file_name}.png")


def _plot_nll_grad(
    validation_results: dict[str, Any],
    file_name: str,
    title: str,
    subfolder: str | None = None,
) -> None:
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

    # Choose markers based on number of steps (no markers if > 10 steps)
    use_markers = len(steps) <= 10
    marker_nll = "o-" if use_markers else "-"
    marker_bd = "s-" if use_markers else "-"
    marker_bp = "^-" if use_markers else "-"
    marker_td = "d-" if use_markers else "-"

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
        ax.plot(
            steps,
            nlls,
            marker_nll,
            linewidth=2,
            color=color,
            label=nll_label,
            alpha=0.8,
        )

        # Plot 2: Gradient Norm - Beta Dust (Uses simple legend)
        ax = axes[0, 1]
        ax.plot(
            steps,
            g_bd,
            marker_bd,
            linewidth=2,
            color=color,
            label=base_label,
            alpha=0.8,
        )

        # Plot 3: Gradient Norm - Beta PL
        ax = axes[1, 0]
        ax.plot(
            steps,
            g_bp,
            marker_bp,
            linewidth=2,
            color=color,
            label=base_label,
            alpha=0.8,
        )

        # Plot 4: Gradient Norm - Temp Dust
        ax = axes[1, 1]
        ax.plot(
            steps,
            g_td,
            marker_td,
            linewidth=2,
            color=color,
            label=base_label,
            alpha=0.8,
        )

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


def plot_gradient_validation(
    validation_results: dict[str, Any] | list[dict[str, Any]],
    file_name: str,
    title: str,
    plot_type: str = "nll-grad",
    labels: list[str] | None = None,
    patches: dict[str, Array] | None = None,
    mask_indices: Array | None = None,
    nside: int | None = None,
    subfolder: str | None = None,
) -> None:
    """
    Dispatch to appropriate plotting function based on plot_type.

    Parameters
    ----------
    validation_results : dict or list of dicts
        Validation results from compute_gradient_validation.
    file_name : str
        Output file name (without extension).
    title : str
        Plot title.
    plot_type : str
        One of: 'nll-grad', 'nll', 'grad', 'grad-maps-{idx}'.
    labels : list of str, optional
        Labels for each result when aggregating multiple runs.
    patches : dict, optional
        Patch indices (required for grad-maps).
    mask_indices : Array, optional
        Mask indices (required for grad-maps).
    nside : int, optional
        HEALPix nside (required for grad-maps).
    subfolder : str, optional
        Subfolder for output.
    """
    if plot_type == "nll-grad":
        # Single result only for nll-grad (original behavior)
        if isinstance(validation_results, list):
            validation_results = validation_results[0]
        _plot_nll_grad(validation_results, file_name, title, subfolder)
    elif plot_type == "nll":
        _plot_nll(validation_results, file_name, title, labels, subfolder)
    elif plot_type == "grad":
        _plot_grad_norms(validation_results, file_name, title, labels, subfolder)
    elif plot_type.startswith("grad-maps-"):
        step_idx = int(plot_type.split("-")[-1])
        if isinstance(validation_results, list):
            validation_results = validation_results[0]
        if patches is None or mask_indices is None or nside is None:
            error("grad-maps requires patches, mask_indices, and nside")
            return
        _plot_grad_maps(
            validation_results, step_idx, patches, mask_indices, nside, file_name, title, subfolder
        )
    else:
        error(f"Unknown plot_type: {plot_type}")


def run_validate(
    matched_results: dict[str, tuple[list[str], int | tuple[int, int], str]],
    names: list[str],
    nside: int,
    instrument: FGBusterInstrument,
    steps: int,
    noise_ratio: float,
    scales: list[float],
    plot_type: str = "nll-grad",
    perturb_beta_dust: str = "all",
    perturb_beta_pl: str = "all",
    perturb_temp_dust: str = "all",
    aggregate: bool = False,
    use_vmap: bool = True,
) -> None:
    """Entry point for 'validate' subcommand.

    Parameters
    ----------
    plot_type : str
        Plot type: 'nll-grad', 'nll', 'grad', or 'grad-maps-{idx}'.
    perturb_beta_dust, perturb_beta_pl, perturb_temp_dust : str
        Perturbation specs: 'all', '-1', '0:30', or '0,1,2'.
    aggregate : bool
        If True, combine all runs on same plot (except grad-maps).
    use_vmap : bool
        If True (default), use vmap. If False, use for-loop (less memory).
    """
    for name, (kw, matched_folders) in zip(names, matched_results.items()):
        folders, run_indices, root_dir = matched_folders

        plot_subfolder = kw
        if root_dir:
            plot_subfolder = os.path.join(root_dir, kw)

        # Normalize run_indices
        if isinstance(run_indices, int):
            run_indices_list = [run_indices]
        elif isinstance(run_indices, tuple) and len(run_indices) == 2:
            run_indices_list = list(range(run_indices[0], run_indices[1] + 1))
        else:
            run_indices_list = list(run_indices)  # type: ignore

        # Collect results for aggregation
        all_val_res = []
        all_labels = []
        last_patches = None
        last_indices = None

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

            for run_idx in run_indices_list:
                info(f"Validating run index {run_idx} in folder '{folder}'")

                # 1. Prepare Data
                run_data_sliced = index_run_data(full_results, run_idx)
                nll = run_data_sliced["NLL"]
                indx = np.argmin(nll)

                patches = {
                    "beta_dust_patches": run_data_sliced["beta_dust_patches"],
                    "beta_pl_patches": run_data_sliced["beta_pl_patches"],
                    "temp_dust_patches": run_data_sliced["temp_dust_patches"],
                }

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
                    perturb_beta_dust=perturb_beta_dust,
                    perturb_beta_pl=perturb_beta_pl,
                    perturb_temp_dust=perturb_temp_dust,
                    use_vmap=use_vmap,
                )

                base_name = os.path.basename(folder.rstrip("/"))

                if aggregate:
                    all_val_res.append(val_res)
                    all_labels.append(f"{base_name}_s{run_idx}")
                    last_patches = patches
                    last_indices = indices
                else:
                    # Plot immediately
                    file_name = f"{base_name}_seed_{run_idx}"
                    plot_gradient_validation(
                        val_res,
                        file_name=file_name,
                        title=name,
                        plot_type=plot_type,
                        patches=patches,
                        mask_indices=indices,
                        nside=nside,
                        subfolder=plot_subfolder,
                    )

        # Aggregated plot
        if aggregate and all_val_res:
            file_name = f"{kw}_aggregated"
            plot_gradient_validation(
                all_val_res,
                file_name=file_name,
                title=name,
                plot_type=plot_type,
                labels=all_labels,
                patches=last_patches,
                mask_indices=last_indices,
                nside=nside,
                subfolder=plot_subfolder,
            )
