import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import operator
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


import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from furax._instruments.sky import get_noise_sigma_from_instrument
from furax.comp_sep import (
    negative_log_likelihood,
    sky_signal,
    spectral_cmb_variance,
)
from furax.obs.landscapes import FrequencyLandscape, HealpixLandscape
from furax.obs.operators import NoiseDiagonalOperator
from jax_grid_search import DistributedGridSearch, ProgressBar, optimize
from jax_healpy import get_clusters, get_cutout_from_mask, normalize_by_first_occurrence
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, simulate_D_from_params
from instruments import get_instrument
from plotting import (
    plot_cmb_nll_vs_B_d_patches_with_noise,
)

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
        "-np",
        "--nb-plot",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Runs to plot",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.2,
        help="Noise ratio",
    )
    parser.add_argument(
        "-av",
        "--average",
        action="store_true",
        help="Average across noise simulations before minimizing",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020",
        choices=MASK_CHOICES,
        help="Mask to use",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument to use",
    )
    parser.add_argument(
        "-b",
        "--best-only",
        action="store_true",
        help="Only generate best results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_folder = f"noise_validation_model{args.mask}_{int(args.noise_ratio * 100)}"
    if args.plot:
        out_folder = f"../results/{out_folder}"
        assert os.path.exists(out_folder), "noise model not found, please run the model first"

        results = np.load(f"{out_folder}/results.npz", allow_pickle=True)
        best_params = np.load(f"{out_folder}/best_params.npz", allow_pickle=True)
        mask = np.load(f"{out_folder}/mask.npy", allow_pickle=True)
        best_params = dict(best_params)
        results = dict(results)
        plot_cmb_nll_vs_B_d_patches_with_noise(results, best_params, out_folder, args.nb_plot)
        # plot_healpix_projection_with_noise(
        #    mask, args.nside, results, best_params, out_folder, args.noise_sim
        # )
        return

    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0
    max_centroids = 300
    n_regions = {
        "temp_dust_patches": 5,
        "beta_dust_patches": 100,
        "beta_pl_patches": 15,
    }
    get_count = lambda c: c if c is not None else 1  # noqa E731
    params_count = {
        "beta_dust": get_count(n_regions["beta_dust_patches"]),
        "temp_dust": get_count(n_regions["temp_dust_patches"]),
        "beta_pl": get_count(n_regions["beta_pl_patches"]),
    }

    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    patch_indices = jax.tree.map(
        lambda c: get_clusters(mask, indices, c, jax.random.key(0), max_centroids=max_centroids),
        n_regions,
    )
    masked_clusters = get_cutout_from_mask(patch_indices, indices)
    # Normalize the cluster to make indexing more logical
    masked_clusters = jax.tree.map(
        lambda g, c: normalize_by_first_occurrence(g, c, max_centroids).astype(jnp.int64),
        masked_clusters,
        n_regions,
    )

    masked_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), masked_clusters)

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
    hp_landscape = HealpixLandscape(nside=nside, stokes="QU")
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    sky = {
        "cmb": hp_landscape.normal(jax.random.key(0)),
        "dust": hp_landscape.normal(jax.random.key(1)),
        "synchrotron": hp_landscape.normal(jax.random.key(2)),
    }
    masked_sky = get_cutout_from_mask(sky, indices)

    params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, params_count)
    params_flat, tree_struct = jax.tree.flatten(params)

    params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].idx), x.shape) * 0.2,
        params_flat,
    )
    best_params = jax.tree.unflatten(tree_struct, params)

    masked_d, masked_fg = simulate_D_from_params(
        best_params,
        masked_clusters,
        nu,
        masked_sky,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
    )
    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    white_noise = f_landscapes.normal(jax.random.key(420)) * noise_ratio
    white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
    sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
    noise = white_noise * sigma
    noised_d = masked_d + noise

    small_n = (sigma * noise_ratio) ** 2
    small_n = 1.0 if noise_ratio == 0 else small_n

    N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

    best_nll = negative_log_likelihood_fn(
        best_params,
        nu=nu,
        d=noised_d,
        N=N,
        patch_indices=masked_clusters,
    )
    best_cmb_var = spectral_cmb_variance_fn(
        best_params,
        nu=nu,
        d=noised_d,
        N=N,
        patch_indices=masked_clusters,
    )

    search_space = {
        "T_d_patches": jnp.arange(5, 21, 5),
        "B_d_patches": jnp.arange(10, 301, 30),
        "B_s_patches": jnp.arange(5, 21, 5),
    }

    max_count = {
        "beta_dust": jnp.max(search_space["B_d_patches"]),
        "temp_dust": jnp.max(search_space["T_d_patches"]),
        "beta_pl": jnp.max(search_space["B_s_patches"]),
    }

    @partial(jax.jit, static_argnums=(5, 6))
    def compute_minimum_variance_with_averaging(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        planck_mask,
        indices,
        max_patches=25,
        progress_bar=None,
    ):
        T_d_patches = T_d_patches.squeeze()
        B_d_patches = B_d_patches.squeeze()
        B_s_patches = B_s_patches.squeeze()

        patch_indices = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }
        patch_indices = jax.tree.map(
            lambda c: get_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=max_centroids
            ),
            patch_indices,
        )
        guess_clusters = get_cutout_from_mask(patch_indices, indices)
        guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)

        def objective_fn(guess_params, nu, d, guess_clusters):
            def single_run(guess_params, nu, masked_d, guess_clusters, noise_id):
                key = jax.random.PRNGKey(noise_id)
                white_noise = f_landscapes.normal(key) * noise_ratio
                white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
                instrument = get_instrument(args.instrument)
                sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
                noise = white_noise * sigma
                noised_d = masked_d + noise

                N = NoiseDiagonalOperator(
                    (sigma * noise_ratio) ** 2, _in_structure=masked_d.structure
                )

                return negative_log_likelihood_fn(
                    guess_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
                )

            nll = jax.vmap(single_run, in_axes=(None, None, None, None, None, 0))(
                guess_params, nu, d, N, guess_clusters, jnp.arange(nb_noise_sim)
            )
            return jnp.mean(nll)

        # lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        # upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        solver = optax.lbfgs()
        final_params, final_state = optimize(
            guess_params,
            objective_fn,
            solver,
            max_iter=200,
            tol=1e-10,
            progress=progress_bar,
            progress_id=0,
            # lower_bound=lower_bound_tree,
            # upper_bound=upper_bound_tree,
            nu=nu,
            d=masked_d,
            guess_clusters=guess_clusters,
            log_updates=True,
        )

        s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters)
        cmb = s["cmb"]
        cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

        cmb_np = jnp.stack([cmb.q, cmb.u])

        nll = negative_log_likelihood_fn(
            final_params, nu=nu, d=masked_d, N=N, patch_indices=guess_clusters
        )

        return {
            "update_history": final_state.update_history,
            "value": cmb_var,
            "CMB_O": cmb_np,
            "NLL": nll,
            "beta_dust": final_params["beta_dust"],
            "temp_dust": final_params["temp_dust"],
            "beta_pl": final_params["beta_pl"],
            "beta_dust_patches": guess_clusters["beta_dust_patches"],
            "temp_dust_patches": guess_clusters["temp_dust_patches"],
            "beta_pl_patches": guess_clusters["beta_pl_patches"],
        }

    @partial(jax.jit, static_argnums=(5, 6))
    def compute_minimum_variance(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        planck_mask,
        indices,
        max_patches=25,
        progress_bar=None,
    ):
        T_d_patches = T_d_patches.squeeze()
        B_d_patches = B_d_patches.squeeze()
        B_s_patches = B_s_patches.squeeze()

        patch_indices = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }
        patch_indices = jax.tree.map(
            lambda c: get_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=max_centroids
            ),
            patch_indices,
        )
        guess_clusters = get_cutout_from_mask(patch_indices, indices)
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

            N = NoiseDiagonalOperator((sigma * noise_ratio) ** 2, _in_structure=masked_d.structure)

            solver = optax.lbfgs()
            final_params, final_state = optimize(
                guess_params,
                negative_log_likelihood_fn,
                solver,
                max_iter=200,
                tol=1e-10,
                progress=progress_bar,
                progress_id=noise_id,
                # lower_bound=lower_bound_tree,
                # upper_bound=upper_bound_tree,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
            )

            s = sky_signal_fn(final_params, nu=nu, d=masked_d, N=N, patch_indices=guess_clusters)
            cmb = s["cmb"]
            cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

            cmb_np = jnp.stack([cmb.q, cmb.u])

            nll = negative_log_likelihood_fn(
                final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
            )

            return {
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

    objective_gridding_fn = (
        compute_minimum_variance_with_averaging if args.average else compute_minimum_variance
    )

    with ProgressBar(*progress_columns) as p:

        @jax.jit
        def objective_function(T_d_patches, B_d_patches, B_s_patches):
            return objective_gridding_fn(
                T_d_patches,
                B_d_patches,
                B_s_patches,
                mask,
                indices,
                max_patches=max_centroids,
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

        if not args.best_only:
            grid_search.run()

    if not args.best_only:
        results = grid_search.stack_results(result_folder=out_folder)
        np.savez(f"{out_folder}/results.npz", **results)

    # Save results
    cmb_map = np.stack([masked_sky["cmb"].q, masked_sky["cmb"].u], axis=0)
    fg_map = np.stack([masked_fg.q, masked_fg.u], axis=1)
    d_map = np.stack([masked_d.q, masked_d.u], axis=1)
    best_params["I_CMB"] = cmb_map
    best_params["I_D"] = d_map
    best_params["I_D_NOCMB"] = fg_map
    best_params["NLL"] = best_nll
    best_params["value"] = best_cmb_var
    best_params["B_d_patches"] = params_count["beta_dust"]
    best_params["T_d_patches"] = params_count["temp_dust"]
    best_params["B_s_patches"] = params_count["beta_pl"]
    best_params["beta_dust_patches"] = masked_clusters["beta_dust_patches"]
    best_params["temp_dust_patches"] = masked_clusters["temp_dust_patches"]
    best_params["beta_pl_patches"] = masked_clusters["beta_pl_patches"]
    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", mask)


if __name__ == "__main__":
    main()
