import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from furax import HomothetyOperator
from furax.obs import negative_log_likelihood, sky_signal
from furax.obs.stokes import Stokes
from jax_grid_search import ProgressBar, optimize
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm import tqdm

from .logging_utils import error, info, success, warning
from .utils import index_run_data


def compute_w(nu, d, patches, max_iter=100):
    """Compute the foreground-only CMB reconstruction (W·d_fg).

    This is a pure computation function with no File I/O.

    Parameters
    ----------
    nu : array_like
        Frequency array in GHz.
    d : Stokes
        Foreground-only frequency maps.
    patches : dict
        Dictionary containing patch indices (beta_dust, temp_dust, beta_pl).
    max_iter : int, optional
        Maximum L-BFGS iterations (default: 100).

    Returns
    -------
    Stokes
        Foreground-only CMB reconstruction (W·d_fg).
    """
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    max_count = {
        "beta_dust": patches["beta_dust_patches"].size,
        "temp_dust": patches["temp_dust_patches"].size,
        "beta_pl": patches["beta_pl_patches"].size,
    }

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }

    guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)

    N = HomothetyOperator(1.0, _in_structure=d.structure)

    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    # Note: We suppress the progress bar for individual runs if running in a huge batch
    # to avoid console spam, or keep it if max_iter is high.
    with ProgressBar(*progress_columns) as p:
        solver = optax.lbfgs()
        final_params, final_state = optimize(
            guess_params,
            negative_log_likelihood_fn,
            solver,
            max_iter=max_iter,
            tol=1e-10,
            progress=p,
            progress_id=0,
            nu=nu,
            N=N,
            d=d,
            patch_indices=patches,
        )

    def W_op(p):
        N = HomothetyOperator(1.0, _in_structure=d.structure)
        return sky_signal(
            p, nu, N, d, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patches
        )["cmb"]

    return W_op(final_params)


def atomic_save_results(result_file, results_dict):
    """Atomically write updated results to disk with backup protection."""
    if os.path.exists(result_file):
        backup_file = result_file.replace(".npz", ".bk.npz")
        os.rename(result_file, backup_file)

    temp_file = result_file.replace(".npz", ".tmp.npz")
    np.savez(temp_file, **results_dict)

    os.rename(temp_file, result_file)


def load_run_data_for_cache(result_folder, run_index):
    """Load necessary data for a specific run index from a result folder."""
    results_path = f"{result_folder}/results.npz"
    best_params_path = f"{result_folder}/best_params.npz"

    try:
        results = dict(np.load(results_path))
        best_params = dict(np.load(best_params_path))
    except (FileNotFoundError, OSError) as e:
        error(f"Failed to load data for {result_folder}: {e}")
        return None, None

    # Slice the specific run data
    run_data = index_run_data(results, run_index)

    fg_stokes_static = Stokes.from_stokes(
        Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
    )

    return run_data, fg_stokes_static


def cache_systematics(matched_results, nside, instrument, force_recompute=False, max_iter=100):
    """Materialise expensive W_D_FG caches for a group of runs.

    This version loads data once per folder and saves once per folder.
    """
    for kw, matched_folders in tqdm(
        matched_results.items(), desc="Preparing folders for caching", unit="group"
    ):
        folders, run_indices = matched_folders

        # Normalize run_indices to list
        if isinstance(run_indices, int):
            run_indices = [run_indices]
        elif isinstance(run_indices, tuple) and len(run_indices) == 2:
            run_indices = list(range(run_indices[0], run_indices[1] + 1))

        for folder in tqdm(
            folders, desc=f"Caching folders matching '{kw}'", unit="folder", leave=False
        ):
            # 1. LOAD PHASE: Load all big files ONCE per folder
            results_path = f"{folder}/results.npz"
            best_params_path = f"{folder}/best_params.npz"

            try:
                # Load full dictionaries into memory
                full_results = dict(np.load(results_path))
                best_params = dict(np.load(best_params_path))
            except (FileNotFoundError, OSError) as e:
                error(f"Failed to load data for {folder}: {e}")
                continue

            fg_stokes_static = Stokes.from_stokes(
                Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
            )

            updates_to_save = {}
            modified = False

            # 2. COMPUTE PHASE: Loop in memory
            for run_index in run_indices:
                cache_key = f"W_D_FG_{run_index}"

                # Check cache in the loaded dictionary
                if cache_key in full_results and not force_recompute:
                    info(f" -> Cache exists for {folder} index {run_index}, skipping.")
                    continue

                info(f" -> Computing {folder} index {run_index}...")

                # Slice the specific run data from the full dictionary
                # This replaces load_run_data_for_cache
                try:
                    # Check bounds
                    first_key = next(iter(full_results.keys()))
                    max_index = len(full_results[first_key]) - 1
                    if run_index > max_index:
                        warning(
                            f"Index {run_index} out of bounds (max {max_index}) for {folder}. Skipping."
                        )
                        continue

                    run_data_sliced = index_run_data(full_results, run_index)
                except Exception as e:
                    error(f"Error slicing data for index {run_index}: {e}")
                    continue

                # Prepare patches
                patches = {
                    "beta_dust_patches": run_data_sliced["beta_dust_patches"],
                    "beta_pl_patches": run_data_sliced["beta_pl_patches"],
                    "temp_dust_patches": run_data_sliced["temp_dust_patches"],
                }

                # Perform pure computation
                W = compute_w(
                    nu=instrument.frequency,
                    d=fg_stokes_static,  # Using the static map loaded once
                    patches=patches,
                    max_iter=max_iter,
                )

                # Store result in memory
                W_numpy = np.stack([W.q, W.u], axis=0)
                updates_to_save[cache_key] = W_numpy
                modified = True

            # 3. SAVE PHASE: Write ONCE per folder
            if modified and updates_to_save:
                info(f"Saving {len(updates_to_save)} new entries to {results_path}...")
                full_results.update(updates_to_save)
                atomic_save_results(results_path, full_results)
                success(f"Updated {folder}")
            else:
                info(f"No changes for {folder}")

    return 0
