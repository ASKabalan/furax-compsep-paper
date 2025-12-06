"""Compute module for r_analysis - handles all result computation and aggregation."""

from collections import OrderedDict

import healpy as hp
import jax.numpy as jnp
import numpy as np
from furax._instruments.sky import get_sky
from furax.obs.stokes import Stokes
from jax_healpy.clustering import combine_masks
from tqdm import tqdm

from .caching import compute_w
from .logging_utils import format_residual_flags, hint, info, warning
from .r_estimate import estimate_r, get_camb_templates
from .residuals import (
    compute_cl_bb_sum,
    compute_cl_obs_bb,
    compute_cl_true_bb,
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)
from .utils import (
    index_run_data,
    params_to_maps,
)


def get_compute_flags(args, snapshot_mode=False):
    """Determine what computations are needed based on args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with visualization toggles.
    snapshot_mode : bool, optional
        If True, all flags are set to True for complete computation (default: False).

    Returns
    -------
    dict
        Dictionary with computation flags:
        - needs_residual_maps: bool
        - needs_residual_spectra: bool
        - needs_r_estimation: bool
        - need_patch_maps: bool
        - need_validation_curves: bool
        - compute_syst: bool
        - compute_stat: bool
        - compute_total: bool
    """
    if snapshot_mode:
        return {
            "needs_residual_maps": True,
            "needs_residual_spectra": True,
            "needs_r_estimation": True,
            "need_patch_maps": True,
            "need_validation_curves": True,
            "compute_syst": True,
            "compute_stat": True,
            "compute_total": True,
        }

    # Extract flags from args with safe getattr for optional attributes
    plot_cmb_recon = getattr(args, "plot_cmb_recon", False)
    plot_systematic_maps = getattr(args, "plot_systematic_maps", False)
    plot_statistical_maps = getattr(args, "plot_statistical_maps", False)
    plot_all = getattr(args, "plot_all", False)
    plot_cl_spectra = getattr(args, "plot_cl_spectra", False)
    plot_all_spectra = getattr(args, "plot_all_spectra", False)
    plot_r_estimation = getattr(args, "plot_r_estimation", False)
    plot_all_r_estimation = getattr(args, "plot_all_r_estimation", False)
    plot_r_vs_c = getattr(args, "plot_r_vs_c", False)
    plot_r_vs_v = getattr(args, "plot_r_vs_v", False)
    plot_illustrations = getattr(args, "plot_illustrations", False)
    plot_params = getattr(args, "plot_params", False)
    plot_patches = getattr(args, "plot_patches", False)
    plot_validation_curves = getattr(args, "plot_validation_curves", False)
    plot_all_metrics = getattr(args, "plot_all_metrics", False)

    needs_residual_maps = (
        plot_cmb_recon or plot_systematic_maps or plot_statistical_maps or plot_all
    )
    needs_residual_spectra = plot_cl_spectra or plot_all_spectra or plot_all
    needs_r_estimation = (
        plot_r_estimation
        or plot_all_r_estimation
        or plot_r_vs_c
        or plot_r_vs_v
        or plot_illustrations
        or plot_all
    )
    need_patch_maps = plot_illustrations or plot_params or plot_patches or plot_all
    need_validation_curves = plot_validation_curves or plot_all or plot_all_metrics

    compute_syst = needs_residual_spectra or needs_residual_maps
    compute_stat = compute_syst or needs_r_estimation
    compute_total = needs_r_estimation

    return {
        "needs_residual_maps": needs_residual_maps,
        "needs_residual_spectra": needs_residual_spectra,
        "needs_r_estimation": needs_r_estimation,
        "need_patch_maps": need_patch_maps,
        "need_validation_curves": need_validation_curves,
        "compute_syst": compute_syst,
        "compute_stat": compute_stat,
        "compute_total": compute_total,
    }


def normalize_indices(indices):
    """Convert indices spec to list of ints.

    Parameters
    ----------
    indices : int or tuple or list
        Index specification:
        - int: single index
        - tuple (start, end): inclusive range
        - list: explicit list of indices

    Returns
    -------
    list of int
        List of indices to process.
    """
    if isinstance(indices, int):
        return [indices]
    elif isinstance(indices, tuple) and len(indices) == 2:
        return list(range(indices[0], indices[1] + 1))
    return list(indices)


def compute_single_folder(
    folder,
    run_index,
    nside,
    instrument,
    flags,
    full_results=None,
    max_iter=100,
):
    """Process a single result folder for a specific run index.

    Parameters
    ----------
    folder : str
        Path to the result folder.
    run_index : int
        Index of the noise realization to process.
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration object.
    flags : dict
        Computation flags from get_compute_flags().
    full_results : dict, optional
        Pre-loaded results.npz contents to avoid reloading (default: None).
    max_iter : int, optional
        Maximum iterations for W computation if not cached (default: 100).

    Returns
    -------
    dict or None
        Dictionary with computed data for this folder/index, or None if failed.
        Keys include: cmb_recon, cmb_true, mask, indices, NLL, wd, params, patches,
        updates_history, value_history.
    """
    # Load data
    results_path = f"{folder}/results.npz"
    best_params_path = f"{folder}/best_params.npz"
    mask_path = f"{folder}/mask.npy"

    try:
        if full_results is None:
            full_results = dict(np.load(results_path))
        best_params = dict(np.load(best_params_path))
        mask = np.load(mask_path)
    except (FileNotFoundError, OSError) as e:
        warning(f"Failed to load data for {folder}: {e}")
        return None

    # Check bounds
    first_key = next(iter(full_results.keys()))
    max_index = len(full_results[first_key]) - 1
    if run_index > max_index:
        warning(f"Index {run_index} out of bounds (max: {max_index}) for {folder}. Skipping.")
        return None

    # Slice the specific run data
    run_data = index_run_data(full_results, run_index)
    (indices,) = jnp.where(mask == 1)

    # Extract CMB data
    cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
    cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])
    NLL = run_data["NLL"]

    # Compute W_D_FG if needed for systematic residuals
    wd = None
    if flags["compute_syst"]:
        cache_key = f"W_D_FG_{run_index}"
        if cache_key in full_results:
            cached_w = full_results[cache_key]
            wd = Stokes.from_stokes(Q=cached_w[0], U=cached_w[1])
        else:
            hint(
                f"Systematics not cached for index {run_index}. "
                "Use 'r_analysis cache -r ...' for faster loading. Computing now..."
            )
            fg_map = Stokes.from_stokes(
                Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
            )
            patches = {
                "beta_dust_patches": run_data["beta_dust_patches"],
                "beta_pl_patches": run_data["beta_pl_patches"],
                "temp_dust_patches": run_data["temp_dust_patches"],
            }
            wd = compute_w(
                nu=instrument.frequency,
                d=fg_map,
                patches=patches,
                max_iter=max_iter,
            )

    # Extract params and patches if needed
    params = None
    patches = None
    if flags["need_patch_maps"]:
        patches = {
            "beta_dust_patches": run_data["beta_dust_patches"],
            "temp_dust_patches": run_data["temp_dust_patches"],
            "beta_pl_patches": run_data["beta_pl_patches"],
        }
        params = {
            "beta_dust": run_data.get("beta_dust"),
            "temp_dust": run_data.get("temp_dust"),
            "beta_pl": run_data.get("beta_pl"),
        }

    # Extract validation curves if needed
    updates_history = None
    value_history = None
    if flags["need_validation_curves"] and "update_history" in run_data:
        updates_history = run_data["update_history"][..., 0]
        value_history = run_data["update_history"][..., 1]

    return {
        "cmb_recon": cmb_recon,
        "cmb_true": cmb_true,
        "mask": mask,
        "indices": indices,
        "NLL": NLL,
        "wd": wd,
        "params": params,
        "patches": patches,
        "updates_history": updates_history,
        "value_history": value_history,
        "run_data": run_data,
        "best_params": best_params,
    }


def compute_group(title, folders, run_indices, nside, instrument, flags, max_iter=100):
    """Process a group of folders for given run indices.

    Parameters
    ----------
    title : str
        Identifier for this run group (for logging).
    folders : list of str
        List of result folder paths.
    run_indices : int or tuple or list
        Index specification for run indices.
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration object.
    flags : dict
        Computation flags from get_compute_flags().
    max_iter : int, optional
        Maximum iterations for W computation if not cached (default: 100).

    Returns
    -------
    tuple or None
        (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data) if successful,
        None if no valid data found.
    """
    if not folders:
        warning(f"No folders provided for '{title}'")
        return None

    indices_spec = normalize_indices(run_indices)

    # Collect data from all folders/indices
    cmb_recons, cmb_maps, masks = [], [], []
    indices_list, w_d_list, NLLs = [], [], []
    params_list, patches_list = [], []
    updates_history_list, value_history_list = [], []

    previous_mask_size = {
        "beta_dust_patches": 0,
        "temp_dust_patches": 0,
        "beta_pl_patches": 0,
    }

    info(
        format_residual_flags(flags["compute_syst"], flags["compute_stat"], flags["compute_total"])
    )

    for folder in tqdm(folders, desc=f"  Folders for {title}", leave=False, unit="folder"):
        # Load results once per folder
        results_path = f"{folder}/results.npz"
        try:
            full_results = dict(np.load(results_path))
        except (FileNotFoundError, OSError) as e:
            warning(f"Failed to load {results_path}: {e}")
            continue

        for run_index in indices_spec:
            result = compute_single_folder(
                folder,
                run_index,
                nside,
                instrument,
                flags,
                full_results=full_results,
                max_iter=max_iter,
            )
            if result is None:
                continue

            cmb_recons.append(result["cmb_recon"])
            cmb_maps.append(result["cmb_true"])
            masks.append(result["mask"])
            indices_list.append(result["indices"])
            NLLs.append(result["NLL"])

            if result["wd"] is not None:
                w_d_list.append(result["wd"])

            if result["params"] is not None and result["patches"] is not None:
                params, patches, previous_mask_size = params_to_maps(
                    result["run_data"], previous_mask_size
                )
                params_list.append(params)
                patches_list.append(patches)

            if result["updates_history"] is not None:
                updates_history_list.append(result["updates_history"])
                value_history_list.append(result["value_history"])

    if len(masks) == 0:
        warning(f"No valid data found for '{title}'. Skipping this run.")
        return None

    # Aggregate results
    full_mask = np.logical_or.reduce(masks)
    f_sky = full_mask.sum() / len(full_mask)

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)

    wd = None
    if flags["compute_syst"] and len(w_d_list) > 0:
        wd = combine_masks(w_d_list, indices_list, nside)

    NLL_summed = np.sum(NLLs, axis=0)

    # Params and patches maps
    if flags["need_patch_maps"] and params_list:
        params_map = combine_masks(params_list, indices_list, nside)
        patches_map = combine_masks(patches_list, indices_list, nside)
    else:
        params_map = None
        patches_map = {
            "beta_dust_patches": np.zeros(hp.nside2npix(nside)),
            "temp_dust_patches": np.zeros(hp.nside2npix(nside)),
            "beta_pl_patches": np.zeros(hp.nside2npix(nside)),
        }

    # CAMB templates for spectra
    needs_camb = flags["needs_residual_spectra"] or flags["needs_r_estimation"]
    if needs_camb:
        ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, _ = get_camb_templates(nside=64)
    else:
        ell_range = cl_bb_r1 = cl_bb_r0 = cl_bb_lens = None

    # Get true sky for residuals
    needs_sky = flags["compute_syst"] or flags["compute_stat"] or flags["compute_total"]
    if needs_sky:
        s_true = get_sky(64, "c1d1s1").components[0].map.value
    else:
        s_true = None

    # Compute residuals
    cl_syst_res, syst_map, cl_stat_res, stat_maps = None, None, None, None

    if flags["compute_syst"] and wd is not None:
        cl_syst_res, syst_map = compute_systematic_res(wd, f_sky, ell_range)
        info(f"Systematic residuals: min={np.min(cl_syst_res):.2e}, max={np.max(cl_syst_res):.2e}")

    if flags["compute_stat"] and flags["compute_syst"] and syst_map is not None:
        cl_stat_res, stat_maps = compute_statistical_res(
            combined_cmb_recon, s_true, f_sky, ell_range, syst_map
        )
        info(f"Statistical residuals: min={np.min(cl_stat_res):.2e}, max={np.max(cl_stat_res):.2e}")

    cl_total_res = None
    if flags["compute_total"]:
        if cl_syst_res is not None and cl_stat_res is not None:
            cl_total_res = cl_syst_res + cl_stat_res
        else:
            cl_total_res, _ = compute_total_res(combined_cmb_recon, s_true, f_sky, ell_range)
        info(f"Total residuals: min={np.min(cl_total_res):.2e}, max={np.max(cl_total_res):.2e}")

    # True Cl
    if ell_range is not None and s_true is not None:
        cl_true = compute_cl_true_bb(s_true, ell_range)
    else:
        cl_true = None

    # Observed Cl
    if flags["compute_total"] and ell_range is not None:
        cl_bb_obs = compute_cl_obs_bb(cl_total_res, cl_bb_lens)
    else:
        cl_bb_obs = None

    # Cl BB sum for illustrations
    if flags["need_patch_maps"] or flags["need_validation_curves"]:
        cl_bb_sum = compute_cl_bb_sum(combined_cmb_recon, f_sky, ell_range)
    else:
        cl_bb_sum = None

    # R estimation
    r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = None, None, None, None, None
    if flags["compute_total"] and flags["needs_r_estimation"] and cl_bb_obs is not None:
        stat_res_for_r = cl_stat_res if cl_stat_res is not None else np.zeros_like(ell_range)
        r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
            cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, stat_res_for_r, f_sky
        )
        info(f"r estimation: {r_best:.4f} +{sigma_r_pos:.4f} -{sigma_r_neg:.4f}")

    # Build output pytrees
    cmb_pytree = {
        "cmb": cmb_stokes,
        "cmb_recon": combined_cmb_recon,
        "patches_map": patches_map,
        "cl_bb_sum": cl_bb_sum,
        "nll_summed": NLL_summed,
    }
    cl_pytree = {
        "cl_bb_r1": cl_bb_r1,
        "cl_bb_r0": cl_bb_r0,
        "cl_true": cl_true,
        "ell_range": ell_range,
        "cl_bb_obs": cl_bb_obs,
        "cl_bb_lens": cl_bb_lens,
        "cl_syst_res": cl_syst_res,
        "cl_total_res": cl_total_res,
        "cl_stat_res": cl_stat_res,
    }
    r_pytree = {
        "r_best": r_best,
        "sigma_r_neg": sigma_r_neg,
        "sigma_r_pos": sigma_r_pos,
        "r_grid": r_grid,
        "L_vals": L_vals,
    }
    residual_pytree = {
        "syst_map": syst_map,
        "stat_maps": stat_maps,
    }
    plotting_data = {
        "params_map": params_map,
        "updates_history": updates_history_list if updates_history_list else None,
        "value_history": value_history_list if value_history_list else None,
    }

    return cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data


def compute_all(
    matched_results,
    nside,
    instrument,
    flags,
    max_iter,
    titles=None,
):
    """Compute results for all matched run groups.

    Parameters
    ----------
    matched_results : dict
        Format: {kw: (folders_list, run_indices)}
        Same format as returned by run_grep() and used by cache_systematics().
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration object.
    flags : dict
        Computation flags from get_compute_flags().
    titles : dict, optional
        Format: {kw: title_string}
        If not provided, uses kw as title.
    max_iter : int, optional
        Maximum iterations for W computation if not cached (default: 100).

    Returns
    -------
    OrderedDict
        Dictionary keyed by title with values:
        (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data)
    """
    if titles is None:
        titles = {}

    results = OrderedDict()

    for kw, (folders, indices) in tqdm(
        matched_results.items(), desc="Processing run groups", unit="group"
    ):
        title = titles.get(kw, kw)
        info(f"Computing results for '{title}' ({len(folders)} folders)")

        result = compute_group(
            title=title,
            folders=folders,
            run_indices=indices,
            nside=nside,
            instrument=instrument,
            flags=flags,
            max_iter=max_iter,
        )

        if result is not None:
            results[title] = result

    return results
