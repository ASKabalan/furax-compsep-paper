import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
from collections import OrderedDict
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from furax._instruments.sky import get_sky
from furax.obs.stokes import Stokes
from jax_healpy.clustering import combine_masks
from tqdm import tqdm

from furax_cs.data.instruments import get_instrument

from .caching import (
    SNAPSHOT_VERSION,
    _tree_to_numpy,
    cache_expensive_computations,
    compute_w,
    load_snapshot,
    save_snapshot_entry,
    write_snapshot_manifest,
)
from .logging_utils import (
    format_cache_summary,
    format_folder_summary,
    format_r_result,
    format_residual_flags,
    info,
    success,
    warning,
)
from .parser import parse_args
from .plotting import (
    plot_all_cl_residuals,
    plot_all_cmb,
    plot_all_r_estimation,
    plot_all_statistical_residuals,
    plot_all_systematic_residuals,
    plot_all_variances,
    plot_cl_residuals,
    plot_cmb_reconstructions,
    plot_params,
    plot_patches,
    plot_r_estimator,
    plot_r_vs_clusters,
    plot_statistical_residual_maps,
    plot_systematic_residual_maps,
    plot_validation_curves,
    plot_variance_vs_clusters,
    plot_variance_vs_r,
    set_font_size,
    set_output_format,
)
from .r_estimate import estimate_r, get_camb_templates
from .residuals import (
    compute_cl_bb_sum,
    compute_cl_obs_bb,
    compute_cl_true_bb,
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)
from .run_management import expand_run_specs, matches_filter, parse_filter_kw
from .utils import index_run_data, params_to_maps

out_folder = "plots/"


def compute_results(name, filtered_results, nside, instrument, args, run_index=0):
    """Aggregate intermediate products needed for plotting a run group.

    Parameters
    ----------
    name : str
        Identifier for this run group.
    filtered_results : list of str
        List of result folder paths matching the run specification.
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration object.
    args : argparse.Namespace
        Command-line arguments controlling which computations to perform.
    run_index : int, optional
        Index of the noise realization to analyze (default: 0).

    Returns
    -------
    tuple or None
        (cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data) if successful,
        None if no valid data found.
    """
    if len(filtered_results) == 0:
        warning(f"No results found matching filter criteria for '{name}'")
        return

    cmb_recons, cmb_maps, masks, NLLs = [], [], [], []
    indices_list, w_d_list = [], []
    params_list, patches_list = [], []
    updates_history, value_history = [], []

    needs_residual_maps = (
        args.plot_cmb_recon
        or args.plot_systematic_maps
        or args.plot_statistical_maps
        or args.plot_all
    )
    needs_residual_spectra = args.plot_cl_spectra or args.plot_all_spectra or args.plot_all
    needs_r_estimation = (
        args.plot_r_estimation
        or args.plot_all_r_estimation
        or args.plot_r_vs_c
        or args.plot_r_vs_v
        or args.plot_illustrations
        or args.plot_all
    )

    compute_syst = args.compute_residuals in ["all", "systematic"]
    compute_stat = args.compute_residuals in ["all", "statistical"]
    compute_total = args.compute_residuals in ["all", "total"]

    if needs_residual_spectra or needs_residual_maps:
        compute_syst = True
        compute_stat = True
    if needs_r_estimation:
        compute_total = True
        compute_stat

    info(format_residual_flags(compute_syst, compute_stat, compute_total))

    needs_camb = (
        args.plot_cl_spectra
        or args.plot_all_spectra
        or args.plot_r_estimation
        or args.plot_all_r_estimation
        or args.plot_r_vs_c
        or args.plot_r_vs_v
        or args.plot_illustrations
        or args.plot_all
    )
    if needs_camb:
        ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, _ = get_camb_templates(nside=64)
    else:
        ell_range = cl_bb_r1 = cl_bb_r0 = cl_bb_lens = None

    previous_mask_size = {
        "beta_dust_patches": 0,
        "temp_dust_patches": 0,
        "beta_pl_patches": 0,
    }

    mem_run_name = None
    mem_run_data = None
    for folder in tqdm(filtered_results, desc=f"  Folders for {name}", leave=False, unit="folder"):
        file = f"{folder}/results.npz"
        if folder == mem_run_name:
            run_data = mem_run_data
        else:
            mem_run_name = folder
            mem_run_data = dict(np.load(file))
            run_data = mem_run_data
        best_params = dict(np.load(f"{folder}/best_params.npz"))
        mask = np.load(f"{folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)
        f_sky = mask.sum() / len(mask)

        first_key = next(iter(run_data.keys()))
        max_index = len(run_data[first_key]) - 1

        if run_index > max_index:
            warning(
                f"Index {run_index} out of bounds (max: {max_index}) for folder {folder}. Skipping."
            )
            continue

        run_data = index_run_data(run_data, run_index)
        NLL = run_data["NLL"]

        cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
        fg_map = Stokes.from_stokes(
            Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
        )
        cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])

        if compute_syst:
            wd = compute_w(
                instrument.frequency,
                fg_map,
                run_data,
                result_file=f"{folder}/results.npz",
                run_index=run_index,
                max_iter=args.max_iterations,
            )
        else:
            wd = None

        if args.plot_illustrations or args.plot_params or args.plot_patches or args.plot_all:
            params, patches, previous_mask_size = params_to_maps(run_data, previous_mask_size)
            params_list.append(params)
            patches_list.append(patches)

        if args.plot_validation_curves or args.plot_all or args.plot_all_metrics:
            updates_history.append(run_data["update_history"][..., 0])
            value_history.append(run_data["update_history"][..., 1])

        cmb_recons.append(cmb_recon)
        cmb_maps.append(cmb_true)
        if wd is not None:
            w_d_list.append(wd)
        masks.append(mask)
        indices_list.append(indices)
        NLLs.append(NLL)

    if len(masks) == 0:
        warning(f"No valid data found for '{name}' with index {run_index}. Skipping this run.")
        return None

    full_mask = np.logical_or.reduce(masks)

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)

    if compute_syst and len(w_d_list) > 0:
        wd = combine_masks(w_d_list, indices_list, nside)
    else:
        wd = None

    NLL_summed = np.sum(NLLs, axis=0)

    if args.plot_illustrations or args.plot_params or args.plot_patches:
        params_map = combine_masks(params_list, indices_list, nside)
        patches_map = combine_masks(patches_list, indices_list, nside)
    else:
        params_map = None
        patches_map = {
            "beta_dust_patches": np.zeros(hp.nside2npix(nside)),
            "temp_dust_patches": np.zeros(hp.nside2npix(nside)),
            "beta_pl_patches": np.zeros(hp.nside2npix(nside)),
        }

    needs_sky = compute_syst or compute_stat or compute_total
    if needs_sky:
        s_true = get_sky(64, "c1d1s1").components[0].map.value
    else:
        s_true = None

    f_sky = full_mask.sum() / len(full_mask)

    cl_syst_res, syst_map, cl_stat_res, stat_maps = None, None, None, None

    if compute_syst and wd is not None:
        cl_syst_res, syst_map = compute_systematic_res(wd, f_sky, ell_range)
        info(f"Systematic residuals: min={np.min(cl_syst_res):.2e}, max={np.max(cl_syst_res):.2e}")

    if compute_stat and compute_syst and syst_map is not None:
        cl_stat_res, stat_maps = compute_statistical_res(
            combined_cmb_recon, s_true, f_sky, ell_range, syst_map
        )
        info(f"Statistical residuals: min={np.min(cl_stat_res):.2e}, max={np.max(cl_stat_res):.2e}")

    if compute_total:
        if cl_syst_res is not None and cl_stat_res is not None:
            cl_total_res = cl_syst_res + cl_stat_res
        else:
            cl_total_res, _ = compute_total_res(combined_cmb_recon, s_true, f_sky, ell_range)
        info(f"Total residuals: min={np.min(cl_total_res):.2e}, max={np.max(cl_total_res):.2e}")
    else:
        cl_total_res = None

    if ell_range is not None and s_true is not None:
        cl_true = compute_cl_true_bb(s_true, ell_range)
    else:
        cl_true = None

    if compute_total and ell_range is not None:
        cl_bb_obs = compute_cl_obs_bb(cl_total_res, cl_bb_lens)
    else:
        cl_bb_obs = None

    if args.plot_illustrations or args.plot_all or args.plot_all_metrics:
        cl_bb_sum = compute_cl_bb_sum(combined_cmb_recon, f_sky, ell_range)
    else:
        cl_bb_sum = None

    if compute_total and needs_r_estimation and cl_bb_obs is not None:
        stat_res_for_r = cl_stat_res if cl_stat_res is not None else np.zeros_like(ell_range)
        r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
            cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, stat_res_for_r, f_sky
        )
        info(f"r estimation: {r_best:.4f} +{sigma_r_pos:.4f} -{sigma_r_neg:.4f}")
    else:
        r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = None, None, None, None, None

    plotting_data = {
        "params_map": params_map,
        "updates_history": updates_history if args.plot_validation_curves else None,
        "value_history": value_history if args.plot_validation_curves else None,
    }

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
        "cl_total_res": cl_total_res if compute_total else None,
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

    return cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data


def plot_results(name, cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data, args):
    """Generate per-run plots according to CLI flags.

    Parameters
    ----------
    name : str
        Run identifier for labeling plots.
    cmb_pytree : dict
        CMB reconstruction data (cmb, cmb_recon, patches_map, etc.).
    cl_pytree : dict
        Power spectra data (cl_bb_r1, cl_syst_res, cl_total_res, etc.).
    r_pytree : dict
        Tensor-to-scalar ratio estimation data (r_best, sigma_r_neg, sigma_r_pos, etc.).
    residual_pytree : dict
        Residual map data (syst_map, stat_maps).
    plotting_data : dict
        Additional plotting data (params_map, updates_history, value_history).
    args : argparse.Namespace
        Command-line arguments controlling which plots to generate.
    """
    cmb_stokes = cmb_pytree["cmb"]
    combined_cmb_recon = cmb_pytree["cmb_recon"]
    patches_map = cmb_pytree["patches_map"]

    cl_bb_r1 = cl_pytree["cl_bb_r1"]
    cl_true = cl_pytree["cl_true"]
    ell_range = cl_pytree["ell_range"]
    cl_bb_obs = cl_pytree["cl_bb_obs"]
    cl_bb_lens = cl_pytree["cl_bb_lens"]
    cl_syst_res = cl_pytree["cl_syst_res"]
    cl_total_res = cl_pytree["cl_total_res"]
    cl_stat_res = cl_pytree["cl_stat_res"]
    cl_bb_r0 = cl_pytree.get("cl_bb_r0")

    r_best = r_pytree["r_best"]
    sigma_r_neg = r_pytree["sigma_r_neg"]
    sigma_r_pos = r_pytree["sigma_r_pos"]
    r_grid = r_pytree["r_grid"]
    L_vals = r_pytree["L_vals"]

    syst_map = residual_pytree.get("syst_map")
    stat_maps = residual_pytree.get("stat_maps")

    params_map = plotting_data.get("params_map")
    updates_history = plotting_data.get("updates_history")
    value_history = plotting_data.get("value_history")

    if (args.plot_params or args.plot_all) and params_map is not None:
        plot_params(name, params_map)
    if (args.plot_patches or args.plot_all) and patches_map is not None:
        plot_patches(name, patches_map)

    if (args.plot_validation_curves or args.plot_all) and updates_history is not None:
        plot_validation_curves(name, updates_history, value_history)

    if (
        (args.plot_cmb_recon or args.plot_all)
        and cmb_stokes is not None
        and combined_cmb_recon is not None
    ):
        plot_cmb_reconstructions(name, cmb_stokes, combined_cmb_recon)

    if (args.plot_systematic_maps or args.plot_all) and syst_map is not None:
        plot_systematic_residual_maps(name, syst_map)

    if (args.plot_statistical_maps or args.plot_all) and stat_maps is not None:
        plot_statistical_residual_maps(name, stat_maps)

    if args.plot_cl_spectra and cl_bb_obs is not None:
        plot_cl_residuals(
            name,
            cl_bb_obs,
            cl_syst_res,
            cl_total_res,
            cl_stat_res,
            cl_bb_r1,
            cl_bb_r0,
            cl_bb_lens,
            cl_true,
            ell_range,
        )

    if args.plot_r_estimation and r_best is not None:
        plot_r_estimator(
            name,
            r_best,
            sigma_r_neg,
            sigma_r_pos,
            r_grid,
            L_vals,
        )


def run_analysis():
    """Entry point for the r_analysis CLI driver.

    This function orchestrates the complete analysis pipeline:
    1. Loads component separation results from disk.
    2. Filters results based on user-specified run specifications.
    3. Computes systematic and statistical residuals, power spectra, and r estimates.
    4. Generates individual and aggregate plots.
    5. Optionally caches expensive computations or saves snapshots.

    The analysis flow is controlled by command-line arguments parsed via :func:`parse_args`.
    """

    args = parse_args()
    set_output_format(args.output_format)
    set_font_size(args.font_size)
    nside = args.nside
    instrument = get_instrument(args.instrument)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False
    # get name of current folder
    result_folders = args.input_results_dir
    results_kw = {}
    for result_folder in result_folders:
        if not os.path.exists(result_folder):
            raise ValueError(f"Results folder '{result_folder}' does not exist.")
        info(f"Loading data from {result_folder}...")
        for root, dirs, files in os.walk(result_folder):
            if not dirs:  # leaf directory (no subdirs)
                info(f" Handling subfolder: {root}")
                name = os.path.basename(root)
                results_kw[root] = name.split("_")

    if args.output_format != "show":
        os.makedirs(out_folder, exist_ok=True)

    if args.plot_all:
        args.plot_cmb_recon = True
        args.plot_cl_spectra = True
        args.plot_all_cmb_recon = True
        args.plot_all_spectra = True
        args.plot_r_estimation = True
        args.plot_all_r_estimation = True
        args.plot_validation_curves = True
        args.plot_illustrations = True

    run_specs = args.runs or []
    title_specs = args.titles or []
    if run_specs and not title_specs:
        title_specs = run_specs
    if len(run_specs) != len(title_specs):
        raise ValueError("Number of titles (--titles) must match number of runs (--runs).")

    expanded_run_groups = expand_run_specs(run_specs, title_specs) if run_specs else []

    snapshot_store = OrderedDict()
    snapshot_path = Path(args.snapshot) if args.snapshot else None
    snapshot_manifest = None
    if snapshot_path is not None:
        entries, snapshot_manifest = load_snapshot(snapshot_path)
        if entries:
            info(f"Loaded {len(entries)} snapshot entries from {snapshot_path}")
        for title, payload in entries:
            snapshot_store[title] = payload

    results_to_plot = []
    titles_to_plot = []
    indices_to_plot = []

    for run_group in expanded_run_groups:
        for filter_expr, run_index, title in run_group:
            filter_groups = parse_filter_kw(filter_expr)
            group = []
            for result_path, res_kw in results_kw.items():
                if matches_filter(res_kw, filter_groups):
                    group.append(result_path)
            if group:
                results_to_plot.append(group)
                titles_to_plot.append(title)
                indices_to_plot.append(run_index)

    if results_to_plot:
        info(format_folder_summary(results_to_plot))
        if snapshot_store:
            info(format_cache_summary(snapshot_store, titles_to_plot))

    if args.cache_only:
        print("=" * 60)
        mode = "FORCE RECOMPUTE" if args.force_cache else "CACHE-ONLY"
        print(f"{mode} MODE: Computing and caching W_D_FG")
        if args.force_cache:
            print("WARNING: Existing cached values will be overwritten")
        print("=" * 60)

        for name, group_results, run_index in zip(titles_to_plot, results_to_plot, indices_to_plot):
            cache_expensive_computations(
                name,
                group_results,
                nside,
                instrument,
                run_index,
                force_recompute=args.force_cache,
                max_iter=args.max_iterations,
            )

        print("=" * 60)
        print("✓ Cache-only mode completed successfully!")
        print("✓ W_D_FG has been cached for all runs")
        print("✓ You can now run plotting commands to use the cached values")
        print("=" * 60)
        return

    stacked_titles = []
    stacked_cmb = []
    stacked_cl = []
    stacked_r = []
    stacked_syst = []
    stacked_stat = []
    run_iterator = zip(titles_to_plot, results_to_plot, indices_to_plot)
    for name, group_results, run_index in tqdm(
        run_iterator, desc="Processing runs", total=len(titles_to_plot), unit="run"
    ):
        if name in snapshot_store:
            success(f"Using cached data for '{name}' from snapshot")
            entry_payload = snapshot_store[name]
            cmb_pytree = entry_payload.get("cmb")
            cl_pytree = entry_payload.get("cl")
            r_pytree = entry_payload.get("r")
            residual_pytree = entry_payload.get("residual", {})
            plotting_data = entry_payload.get("plotting_data", {})
        else:
            result = compute_results(name, group_results, nside, instrument, args, run_index)
            if result is None:
                continue
            cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data = result
            entry_payload = {
                "cmb": cmb_pytree,
                "cl": cl_pytree,
                "r": r_pytree,
                "residual": residual_pytree,
                "plotting_data": plotting_data,
            }
            snapshot_store[name] = entry_payload
            if snapshot_path is not None:
                if snapshot_manifest is None:
                    snapshot_manifest = {"version": SNAPSHOT_VERSION, "entries": []}
                serializable_entry = {
                    "cmb": _tree_to_numpy(cmb_pytree),
                    "cl": _tree_to_numpy(cl_pytree),
                    "r": _tree_to_numpy(r_pytree),
                    "residual": _tree_to_numpy(residual_pytree),
                    "plotting_data": _tree_to_numpy(plotting_data),
                }
                snapshot_manifest = save_snapshot_entry(
                    snapshot_path, snapshot_manifest, name, serializable_entry
                )
                write_snapshot_manifest(snapshot_path, snapshot_manifest)
            r_msg = format_r_result(
                r_pytree.get("r_best"), r_pytree.get("sigma_r_neg"), r_pytree.get("sigma_r_pos")
            )
            if r_msg:
                success(f"{name} complete ({r_msg})")
            else:
                success(f"{name} complete")

        needs_individual_plots = (
            args.plot_params
            or args.plot_patches
            or args.plot_validation_curves
            or args.plot_cmb_recon
            or args.plot_systematic_maps
            or args.plot_statistical_maps
            or args.plot_cl_spectra
            or args.plot_r_estimation
        )
        if needs_individual_plots:
            plot_results(
                name, cmb_pytree, cl_pytree, r_pytree, residual_pytree, plotting_data, args
            )

        plt.close("all")

        payload_complete = (
            isinstance(cmb_pytree, dict)
            and isinstance(cl_pytree, dict)
            and isinstance(r_pytree, dict)
        )
        if not payload_complete:
            warning(f"Snapshot entry '{name}' is missing required data, skipping aggregation.")
            continue

        stacked_titles.append(name)
        stacked_cmb.append(cmb_pytree)
        stacked_cl.append(cl_pytree)
        stacked_r.append(r_pytree)

        if isinstance(residual_pytree, dict):
            if residual_pytree.get("syst_map") is not None:
                stacked_syst.append(residual_pytree["syst_map"])
            if residual_pytree.get("stat_maps") is not None:
                stacked_stat.append(residual_pytree["stat_maps"])

    if stacked_titles:
        if args.plot_illustrations or args.plot_r_vs_c:
            plot_r_vs_clusters(stacked_titles, stacked_cmb, stacked_r)
        if args.plot_illustrations or args.plot_v_vs_c:
            plot_variance_vs_clusters(stacked_titles, stacked_cmb)
        if args.plot_illustrations or args.plot_r_vs_v:
            plot_variance_vs_r(stacked_titles, stacked_cmb, stacked_r)
        plt.close("all")
    if args.plot_all_cmb_recon and stacked_titles:
        plot_all_cmb(stacked_titles, stacked_cmb)
        plt.close("all")
    if args.plot_all_spectra and stacked_titles:
        plot_all_cl_residuals(stacked_titles, stacked_cl)
        if (args.plot_all_systematic_maps or args.plot_all_spectra) and len(stacked_syst) > 0:
            plot_all_systematic_residuals(stacked_titles, stacked_syst)
        if (args.plot_all_statistical_maps or args.plot_all_spectra) and len(stacked_stat) > 0:
            plot_all_statistical_residuals(stacked_titles, stacked_stat)
        plt.close("all")
    if args.plot_all_r_estimation and stacked_titles:
        plot_all_r_estimation(stacked_titles, stacked_r)
        plt.close("all")
    if args.plot_all_metrics and stacked_titles:
        plot_all_variances(stacked_titles, stacked_cmb)
        plt.close("all")


def main():
    """CLI entry point for the r_analysis tool.

    This is the primary entry point registered as the ``r_analysis`` console script
    in pyproject.toml. It simply invokes :func:`run_analysis`.
    """
    run_analysis()


if __name__ == "__main__":
    run_analysis()
