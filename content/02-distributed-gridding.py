import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
from furax import Config, HomothetyOperator
from furax.comp_sep import (
    negative_log_likelihood,
    spectral_cmb_variance,
)
from furax.obs.landscapes import Stokes
from jax_grid_search import DistributedGridSearch, optimize
from jax_healpy import get_clusters, get_cutout_from_mask

from generate_maps import load_from_cache, save_to_cache


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
        "-c",
        "--cache-run",
        action="store_true",
        help="Run the cache generation step",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    GAL020 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL020"]
    #GAL040 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL040"]
    #GAL060 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL060"]

    nside = args.nside
    #npixel = 12 * nside**2

    if args.cache_run:
        save_to_cache(nside, sky="c1d1s1", noise=True)
        return

    nu, freq_maps = load_from_cache(nside, sky="c1d1s1", noise=True)
    # Check the shape of freq_maps
    print("freq_maps shape:", freq_maps.shape)

    (indices,) = jnp.where(GAL020 == 1)
    freq_maps_m = np.zeros((freq_maps.shape[0], freq_maps.shape[1], len(indices)))
    for i, _ in enumerate(freq_maps):
        freq_maps_m[i, 1] = get_cutout_from_mask(freq_maps[i, 1], indices)

    d = Stokes.from_stokes(Q=freq_maps_m[:, 1, :], U=freq_maps_m[:, 2, :])

    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0

    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    N = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    solver = optax.lbfgs()

    inverser_options = {
        "solver": lx.CG(rtol=1e-6, atol=1e-6, max_steps=1000),
        "solver_throw": False,
    }

    @partial(jax.jit, static_argnums=(5))
    def compute_minimum_variance(
        T_d_patches, B_d_patches, B_s_patches, planck_mask, indices, max_patches=25
    ):
        temp_dust_patch_indices = get_clusters(
            planck_mask,
            indices,
            T_d_patches,
            jax.random.PRNGKey(0),
            max_centroids=max_patches,
        )
        beta_dust_patch_indices = get_clusters(
            planck_mask,
            indices,
            B_d_patches,
            jax.random.PRNGKey(0),
            max_centroids=max_patches,
        )
        beta_pl_patch_indices = get_clusters(
            planck_mask,
            indices,
            B_s_patches,
            jax.random.PRNGKey(0),
            max_centroids=max_patches,
        )

        params = {
            "beta_dust": jnp.full((max_patches,), 1.54),
            "temp_dust": jnp.full((max_patches,), 20.0),
            "beta_pl": jnp.full((max_patches,), (-3.0)),
        }

        patch_indices = {
            "temp_dust_patches": temp_dust_patch_indices,
            "beta_dust_patches": beta_dust_patch_indices,
            "beta_pl_patches": beta_pl_patch_indices,
        }

        masked_clusters = jax.tree.map(
            lambda full_map: get_cutout_from_mask(full_map, indices).astype(jnp.int32),
            patch_indices,
        )

        with Config(**inverser_options):
            final_params, final_state = optimize(
                params,
                negative_log_likelihood_fn,
                solver,
                max_iter=500,
                tol=1e-8,
                verbose=False,
                nu=nu,
                N=N,
                d=d,
                patch_indices=masked_clusters,
            )

        value = spectral_cmb_variance_fn(
            final_params, nu=nu, d=d, N=N, patch_indices=masked_clusters
        )

        return {
            "value": value,
            "beta_dust": final_params["beta_dust"],
            "temp_dust": final_params["temp_dust"],
            "beta_pl": final_params["beta_pl"],
        }

    max_centroids = 1000
    search_space = {
        "T_d_patches": jnp.array([10, 500, 1000]),
        "B_d_patches": jnp.array([10, 250, 500, 750, 1000]),
        "B_s_patches": jnp.array([10, 500, 1000]),
    }
    search_space = {
        "T_d_patches": jnp.array([10, 1000]),
        "B_d_patches": jnp.array([10, 250, 1000]),
        "B_s_patches": jnp.array([10, 1000]),
    }

    @jax.jit
    def objective_function(T_d_patches, B_d_patches, B_s_patches):
        return compute_minimum_variance(
            T_d_patches,
            B_d_patches,
            B_s_patches,
            GAL020,
            indices,
            max_patches=max_centroids,
        )

    # Put the good values for the grid search

    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=1,
        progress_bar=True,
        log_every=0.1,
        result_dir="c1d1s1_search",
    )

    grid_search.run()

    results = grid_search.stack_results(result_folder="c1d1s1_search")
    # Save results
    np.savez("results.npz", **results)


if __name__ == "__main__":
    main()
