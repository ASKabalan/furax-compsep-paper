import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import os
import sys
from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import optax
from furax._instruments.sky import FGBusterInstrument, get_noise_sigma_from_instrument
from furax.comp_sep import (
    negative_log_likelihood,
    spectral_cmb_variance,
)
from furax.obs.landscapes import FrequencyLandscape, HealpixLandscape
from furax.obs.operators import NoiseDiagonalOperator
from jax_grid_search import DistributedGridSearch, optimize , ProgressBar
from jax_healpy import from_cutout_to_fullmap, get_clusters, get_cutout_from_mask
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
from generate_maps import simulate_D_from_params


def filter_constant_param(input_dict, indx):
    out_dict = {}
    out_dict["B_d_patches"] = input_dict["B_d_patches"][indx]
    out_dict["T_d_patches"] = input_dict["T_d_patches"][indx]
    out_dict["B_s_patches"] = input_dict["B_s_patches"][indx]
    out_dict["value"] = input_dict["value"][indx]
    out_dict["NLL"] = input_dict["NLL"][indx]

    out_dict["beta_dust"] = input_dict["beta_dust"][indx][: out_dict["B_d_patches"]]
    out_dict["temp_dust"] = input_dict["temp_dust"][indx][: out_dict["T_d_patches"]]
    out_dict["beta_pl"] = input_dict["beta_pl"][indx][: out_dict["B_s_patches"]]
    return out_dict


def plot_cmb_nll_vs_B_d_patches(results, best_params):
    import matplotlib.pyplot as plt
    import seaborn as sns

    validation_model_folder = "validation_model"
    sns.set_context("paper")
    # Extract values from the grid search results
    B_d_patches = results["B_d_patches"]  # dust patch count (x-axis)
    cmb_variance = results["value"]  # CMB variance values from grid search
    nll = results["NLL"]  # Negative log-likelihood values

    # Create subplots: one for CMB variance, one for NLL
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot CMB variance vs. B_d_patches
    axs[0].scatter(B_d_patches, cmb_variance, color="blue", label="Grid Search")
    axs[0].axhline(
        y=best_params["value"], color="red", linestyle="--", label="Best CMB Variance"
    )
    axs[0].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[0].set_xlabel("B_d_patches (dust patch count)")
    axs[0].set_ylabel("CMB Variance")
    axs[0].set_title("CMB Variance vs. B_d_patches")
    axs[0].legend()

    # Plot NLL vs. B_d_patches
    axs[1].scatter(B_d_patches, nll, color="green", label="Grid Search")
    axs[1].axhline(y=best_params["NLL"], color="red", linestyle="--", label="Best NLL")
    axs[1].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[1].set_xlabel("B_d_patches (dust patch count)")
    axs[1].set_ylabel("Negative Log Likelihood")
    axs[1].set_title("NLL vs. B_d_patches")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{validation_model_folder}/cmb_nll_vs_B_d_patches.png",
        dpi=600,
        transparent=True,
    )


def plot_healpix_projection(mask, nside, results, best_params):
    import matplotlib.pyplot as plt
    import seaborn as sns

    validation_model_folder = "validation_model"
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)
    best_run = filter_constant_param(results, 0)
    patches = get_clusters(
        mask,
        indices,
        best_run["B_d_patches"],
        jax.random.key(0),
        max_centroids=best_run["B_d_patches"],
    ).astype(jnp.int32)
    patches = get_cutout_from_mask(patches, indices)
    best_spectral_params = best_params["beta_dust"][patches]
    result_spectral_params = best_run["beta_dust"][patches]

    best_healpix_map = from_cutout_to_fullmap(best_spectral_params, indices, nside)
    result_healpix_map = from_cutout_to_fullmap(result_spectral_params, indices, nside)

    # Plot the best and result maps
    plt.figure(figsize=(15, 5))
    hp.mollview(
        best_healpix_map, title="Best Beta Dust map", sub=(1, 2, 1), bgcolor=(0.0,) * 4
    )
    hp.mollview(
        result_healpix_map,
        title="Result Beta Dust map",
        sub=(1, 2, 2),
        bgcolor=(0.0,) * 4,
    )
    plt.savefig(
        f"{validation_model_folder}/best_result_healpix_projection.png",
        dpi=600,
        transparent=True,
    )


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
        default=20,
        help="Number of noise simulations",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    GAL020 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL020"]

    if args.plot:
        validation_model_folder = "validation_model"
        assert os.path.exists(validation_model_folder), (
            "Validation model not found, please run the model first"
        )

        results = np.load("results.npz")
        best_params = np.load("best_params.npz")
        best_params = dict(best_params)
        results = dict(results)
        plot_cmb_nll_vs_B_d_patches(results, best_params)
        plot_healpix_projection(GAL020, args.nside, results, best_params)
        return

    nside = args.nside
    nb_noise_sim = args.noise_sim
    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
    max_centroids = 300
    patch_indices = {
        "temp_dust_patches": None,
        "beta_dust_patches": 10,
        "beta_pl_patches": None,
    }
    get_count = lambda c: c if c is not None else 1  # noqa E731
    params_count = {
        "beta_dust": get_count(patch_indices["beta_dust_patches"]),
        "temp_dust": get_count(patch_indices["temp_dust_patches"]),
        "beta_pl": get_count(patch_indices["beta_pl_patches"]),
    }

    mask = GAL020
    (indices,) = jnp.where(mask == 1)

    patch_indices = jax.tree.map(
        lambda c: get_clusters(
            mask, indices, c, jax.random.key(0), max_centroids=max_centroids
        ),
        patch_indices,
    )
    masked_clusters = get_cutout_from_mask(patch_indices, indices)
    masked_clusters = jax.tree.map(lambda x: x.astype(jnp.int32), masked_clusters)

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }
    lower_bound = {
        "beta_dust": 1.0,
        "temp_dust": 10.0,
        "beta_pl": -5.0,
    }
    upper_bound = {
        "beta_dust": 3.0,
        "temp_dust": 30.0,
        "beta_pl": 0.0,
    }

    instrument = FGBusterInstrument.default_instrument()
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
        lambda path, x: x
        + jax.random.normal(jax.random.key(path[0].idx), x.shape) * 0.2,
        params_flat,
    )
    best_params = jax.tree.unflatten(tree_struct, params)

    masked_d = simulate_D_from_params(
        best_params,
        masked_clusters,
        nu,
        masked_sky,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
    )

    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    white_noise = f_landscapes.normal(jax.random.key(420))
    white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
    sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
    noise = white_noise * sigma
    noised_d = masked_d + noise
    N = NoiseDiagonalOperator(sigma**2, _in_structure=masked_d.structure)

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
        "T_d_patches": jnp.array([1]),
        "B_d_patches": jnp.arange(10, 301, 10),
        "B_s_patches": jnp.array([1]),
    }

    max_count = {
        "beta_dust": jnp.max(search_space["B_d_patches"]),
        "temp_dust": jnp.max(search_space["T_d_patches"]),
        "beta_pl": jnp.max(search_space["B_s_patches"]),
    }

    @partial(jax.jit, static_argnums=(5 , 6))
    def compute_minimum_variance(
        T_d_patches, B_d_patches, B_s_patches, planck_mask, indices, max_patches=25 , progress_bar=None
    ):
        def single_run(noise_id):
            key = jax.random.PRNGKey(noise_id)
            white_noise = f_landscapes.normal(key)
            white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
            sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
            noise = white_noise * sigma
            noised_d = masked_d + noise

            N = NoiseDiagonalOperator(sigma**2, _in_structure=masked_d.structure)

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
            guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int32), guess_clusters)

            guess_params = jax.tree.map(
                lambda v, c: jnp.full((c,), v), base_params, max_count
            )
            lower_bound_tree = jax.tree.map(
                lambda v, c: jnp.full((c,), v), lower_bound, max_count
            )
            upper_bound_tree = jax.tree.map(
                lambda v, c: jnp.full((c,), v), upper_bound, max_count
            )

            solver = optax.lbfgs()
            final_params, final_state = optimize(
                guess_params,
                negative_log_likelihood_fn,
                solver,
                max_iter=200,
                tol=1e-10,
                progress=progress_bar,
                progress_id=noise_id,
                lower_bound=lower_bound_tree,
                upper_bound=upper_bound_tree,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
            )

            cmb_var = spectral_cmb_variance_fn(
                final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
            )
            nll = negative_log_likelihood_fn(
                final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters
            )

            return {
                "value": cmb_var,
                "NLL": nll,
                "beta_dust": final_params["beta_dust"],
                "temp_dust": final_params["temp_dust"],
                "beta_pl": final_params["beta_pl"],
            }

        return jax.vmap(single_run)(jnp.arange(nb_noise_sim))

    # Put the good values for the grid search
    if os.path.exists("noise_validation_model"):
        old_results = DistributedGridSearch.stack_results(
            result_folder="noise_validation_model"
        )
    else:
        old_results = None

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    #with ProgressBar(*progress_columns) as p:
    @jax.jit
    def objective_function(T_d_patches, B_d_patches, B_s_patches):
        return compute_minimum_variance(
            T_d_patches,
            B_d_patches,
            B_s_patches,
            mask,
            indices,
            max_patches=max_centroids,
            progress_bar=None
        )

    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=4,
        progress_bar=True,
        result_dir="noise_validation_model",
        old_results=old_results,
    )

    grid_search.run()

    results = grid_search.stack_results(result_folder="noise_validation_model")

    # Save results
    best_params["NLL"] = best_nll
    best_params["value"] = best_cmb_var
    best_params["B_d_patches"] = params_count["beta_dust"]
    best_params["T_d_patches"] = params_count["temp_dust"]
    best_params["B_s_patches"] = params_count["beta_pl"]
    np.savez("noise_validation_model/results.npz", **results)
    np.savez("noise_validation_model/best_params.npz", **best_params)


if __name__ == "__main__":
    main()
