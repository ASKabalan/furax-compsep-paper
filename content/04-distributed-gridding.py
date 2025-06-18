import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import os
import sys
from functools import partial

import jax

# =============================================================================
# 1. If running on a distributed system, initialize JAX distributed
# =============================================================================
if (
    int(os.environ.get("SLURM_NTASKS", 0)) > 1
    or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
):
    os.environ["VSCODE_PROXY_URI"] = ""
    os.environ["no_proxy"] = ""
    os.environ["NO_PROXY"] = ""
    del os.environ["VSCODE_PROXY_URI"]
    del os.environ["no_proxy"]
    del os.environ["NO_PROXY"]
    jax.distributed.initialize()
# =============================================================================
import glob
import operator

import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from furax._instruments.sky import (
    get_noise_sigma_from_instrument,
)
from furax.comp_sep import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_grid_search import DistributedGridSearch, ProgressBar, optimize
from jax_healpy import get_clusters, get_cutout_from_mask, normalize_by_first_occurrence
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, load_cmb_map, load_fg_map, load_from_cache
from instruments import get_instrument
from plotting import plot_grid_search_results

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="The nside of the map",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=50,
        help="Number of noise simulations",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.2,
        help="Noise ratio",
    )
    parser.add_argument(
        "-tag",
        "--tag",
        type=str,
        default="c1d1s1",
        help="Tag for the observation",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020_U",
        choices=MASK_CHOICES,
        help="Mask to use",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
    )
    parser.add_argument(
        "-b",
        "--best-only",
        action="store_true",
        help="Only generate best results",
    )
    parser.add_argument(
        "-c",
        "--clean-up",
        type=str,
        nargs="?",
        default=None,
        help="Clean up the output folder",
    )
    return parser.parse_args()


def clean_up(folder):
    batch_size = 400
    nb_to_keep = 10

    final_folders = glob.glob("final/*")
    final_folders = [os.path.basename(f) for f in final_folders]
    print(f"Final folders: {final_folders}")

    if not os.path.isdir(folder):
        print(f"Provided folder '{folder}' is not a directory.")
        return

    if folder in final_folders:
        print(f"Folder {folder} already processed, skipping.")
        return

    print(f"Processing folder: {folder}")

    nb_batches = DistributedGridSearch.get_num_batches(folder, batch_size)
    print(f"Number of batches: {nb_batches}")

    stacked_results = None

    for batch_idx in range(nb_batches):
        print(f"Processing batch {batch_idx + 1}/{nb_batches}")

        batch_results = DistributedGridSearch.stack_results(
            folder, batch_size=batch_size, batch_index=batch_idx
        )
        if batch_results is None:
            print(f"Batch {batch_idx} is empty, skipping.")
            continue

        NLL = batch_results["NLL"]
        print(f"NLL dtype: {NLL.dtype}, type(NLL): {type(NLL)}")

        finite_nll = np.isfinite(NLL).all(axis=1)
        if finite_nll.sum() == 0:
            print(f"No finite results in batch {batch_idx}, skipping.")
            continue

        results_finite = {k: v[finite_nll] for k, v in batch_results.items()}
        print(f"Finite results in batch {batch_idx}: {results_finite['NLL'].shape[0]}")

        # Take only the first finite sample, keep axis
        tosave = {k: v[:nb_to_keep] for k, v in results_finite.items()}

        if stacked_results is None:
            stacked_results = {k: [v] for k, v in tosave.items()}
        else:
            for k, v in tosave.items():
                stacked_results[k].append(v)

    if stacked_results is None:
        print(f"No finite results found for folder {folder}.")
        return

    # Stack across batches
    final_results = {k: np.concatenate(v, axis=0) for k, v in stacked_results.items()}

    # sort w.r.t to value
    sorted_indices = np.argsort(
        final_results["value"].mean(axis=tuple(range(1, final_results["value"].ndim)))
    )
    sorted_results = {key: value[sorted_indices] for key, value in final_results.items()}

    output_folder = os.path.join("final", folder)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "results.npz")

    np.savez(output_path, **sorted_results)
    print(f"Saved stacked results to {output_path}")


def main():
    args = parse_args()

    out_folder = f"compsep_{args.tag}_{args.instrument}_{args.mask}_{int(args.noise_ratio * 100)}"
    if args.plot:
        assert os.path.exists(out_folder), "output not found, please run the model first"
        results = np.load(f"{out_folder}/results.npz")
        plot_grid_search_results(
            results, out_folder, best_metric="value", nb_best=12, noise_runs=args.noise_sim
        )
        return

    if args.clean_up is not None:
        clean_up(args.clean_up)
        return

    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }
    # lower_bound = {
    #    "beta_dust": 0.5,
    #    "temp_dust": 6.0,
    #    "beta_pl": -7.0,
    # }
    # upper_bound = {
    #    "beta_dust": 5.0,
    #    "temp_dust": 40.0,
    #    "beta_pl": -0.5,
    # }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    _, freqmaps = load_from_cache(nside, noise=False, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)
    masked_fg = get_cutout_from_mask(fg_stokes, indices, axis=1)
    masked_cmb = get_cutout_from_mask(cmb_map_stokes, indices)

    search_space = {
        "T_d_patches": jnp.array([1, 5, 20, 50, 80, 100, 500, 1000, 2000, 5000, 10000]),
        "B_d_patches": jnp.arange(5000, 10001, 1000),
        "B_s_patches": jnp.array([1, 5, 20, 50, 80, 100, 500, 1000, 2000, 5000, 10000]),
    }

    search_space = jax.tree.map(lambda x: x[x < indices.size], search_space)

    max_count = {
        "beta_dust": np.max(np.array(search_space["B_d_patches"])),
        "temp_dust": np.max(np.array(search_space["T_d_patches"])),
        "beta_pl": np.max(np.array(search_space["B_s_patches"])),
    }
    max_patches = {
        "temp_dust_patches": max_count["temp_dust"],
        "beta_dust_patches": max_count["beta_dust"],
        "beta_pl_patches": max_count["beta_pl"],
    }

    @partial(jax.jit, static_argnums=(5))
    def compute_minimum_variance(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        indices,
        progress_bar=None,
    ):
        T_d_patches = T_d_patches.squeeze()
        B_d_patches = B_d_patches.squeeze()
        B_s_patches = B_s_patches.squeeze()

        n_regions = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }

        patch_indices = jax.tree.map(
            lambda c, mp: get_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=mp, initial_sample_size=1
            ),
            n_regions,
            max_patches,
        )
        guess_clusters = get_cutout_from_mask(patch_indices, indices)
        # Normalize the cluster to make indexing more logical
        guess_clusters = jax.tree.map(
            lambda g, c, mp: normalize_by_first_occurrence(g, c, mp).astype(jnp.int64),
            guess_clusters,
            n_regions,
            max_patches,
        )
        guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        # lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        # upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        def single_run(noise_id):
            key = jax.random.PRNGKey(noise_id)
            white_noise = f_landscapes.normal(key) * noise_ratio
            white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
            instrument = get_instrument(args.instrument)
            sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
            noise = white_noise * sigma
            noised_d = masked_d + noise

            small_n = (sigma * noise_ratio) ** 2
            small_n = 1.0 if noise_ratio == 0 else small_n

            N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

            solver = optax.lbfgs()
            opt = optax.chain(optax.zero_nans(), solver)
            final_params, final_state = optimize(
                guess_params,
                negative_log_likelihood_fn,
                opt,
                max_iter=1000,
                tol=1e-10,
                progress=progress_bar,
                progress_id=noise_id,
                # lower_bound=lower_bound_tree,
                # upper_bound=upper_bound_tree,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
                log_updates=True,
            )

            s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters)
            cmb = s["cmb"] 
            # Variance of the CMB map
            cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))
            # This is equivalent to jnp.var(cmb.q) + jnp.var(cmb.u)

            cmb_np = jnp.stack([cmb.q, cmb.u])

            nll = negative_log_likelihood_fn(
                final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
            )

            return {
                "update_history": final_state.update_history,
                "value": cmb_var,
                "CMB_O": cmb_np,
                "NLL": nll,
                "beta_dust": final_params["beta_dust"],
                "temp_dust": final_params["temp_dust"],
                "beta_pl": final_params["beta_pl"],
            }

        results = jax.vmap(single_run)(jnp.arange(nb_noise_sim))
        results["beta_dust_patches"] = guess_clusters["beta_dust_patches"]
        results["temp_dust_patches"] = guess_clusters["temp_dust_patches"]
        results["beta_pl_patches"] = guess_clusters["beta_pl_patches"]
        return results

    # Put the good values for the grid search
    if os.path.exists(out_folder):
        old_results = DistributedGridSearch.stack_results(result_folder=out_folder)
    else:
        old_results = None

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with ProgressBar(*progress_columns) as p:

        @jax.jit
        def objective_function(T_d_patches, B_d_patches, B_s_patches):
            return compute_minimum_variance(
                T_d_patches,
                B_d_patches,
                B_s_patches,
                indices,
                progress_bar=p,
            )

        grid_search = DistributedGridSearch(
            objective_function,
            search_space,
            batch_size=1,
            progress_bar=True,
            result_dir=out_folder,
            old_results=old_results,
        )
        print(f"Number of combinations: {grid_search.n_combinations}")
        if not args.best_only:
            grid_search.run()

    if not args.best_only:
        results = grid_search.stack_results(result_folder=out_folder)
        np.savez(f"{out_folder}/results.npz", **results)

    # Save results and mask
    best_params = {}
    cmb_map = np.stack([masked_cmb.q, masked_cmb.u], axis=0)
    fg_map = np.stack([masked_fg.q, masked_fg.u], axis=1)
    d_map = np.stack([masked_d.q, masked_d.u], axis=1)
    best_params["I_CMB"] = cmb_map
    best_params["I_D"] = d_map
    best_params["I_D_NOCMB"] = fg_map

    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", mask)
    print("Run complete. Results saved to", out_folder)


if __name__ == "__main__":
    main()
