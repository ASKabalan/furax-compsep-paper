import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
import optax
from furax import Config, HomothetyOperator
from furax._instruments.sky import FGBusterInstrument
from furax.comp_sep import (
    negative_log_likelihood,
    spectral_cmb_variance,
    spectral_log_likelihood,
)
from furax.obs.landscapes import FrequencyLandscape, Stokes
from furax.obs.stokes import Stokes
from generate_maps import simulate_D_from_params
from jax_grid_search import DistributedGridSearch, optimize
from jax_healpy import from_cutout_to_fullmap, get_clusters, get_cutout_from_mask


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
    return parser.parse_args()


def main():
    args = parse_args()

    GAL020 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL020"]
    GAL040 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL040"]
    GAL060 = np.load("../data/masks/GAL_PlanckMasks_64.npz")["GAL060"]

    nside = args.nside
    npixel = 12 * nside**2
    patch_counts = {
        "temp_dust_patches": 1,
        "beta_dust_patches": 100,
        "beta_pl_patches": 1,
    }

    max_centroids = 300
    mask = GAL020
    (indices,) = jnp.where(mask == 1)

    patch_indices = jax.tree.map(
        lambda c: get_clusters(
            mask, indices, c, jax.random.PRNGKey(0), max_centroids=max_centroids
        ),
        patch_counts,
    )
    masked_clusters = jax.tree.map(
        lambda full_map: get_cutout_from_mask(full_map, indices).astype(jnp.int32),
        patch_indices,
    )

    nu = FGBusterInstrument.default_instrument().frequency
    land_scape = FrequencyLandscape(nside=nside, frequencies=nu, stokes="QU")

    sky = {
        "cmb": land_scape.normal(jax.random.key(0)),
        "dust": land_scape.normal(jax.random.key(1)),
        "synchrotron": land_scape.normal(jax.random.key(2)),
    }
    masked_sky = jax.tree.map(
        lambda full_map: get_cutout_from_mask(full_map, indices, axis=1), sky
    )

    best_params = {
        "temp_dust": jnp.full((patch_counts["temp_dust_patches"],), 20.0),
        "beta_dust": jnp.full((patch_counts["beta_dust_patches"],), 1.54),
        "beta_pl": jnp.full((patch_counts["beta_pl_patches"],), -3.0),
    }

    best_params_flat, tree_struct = jax.tree.flatten(best_params)
    best_params = jax.tree.map_with_path(
        lambda path, x: x
        + jax.random.normal(jax.random.key(path[0].idx), x.shape) * 0.2,
        best_params_flat,
    )
    best_params = jax.tree.unflatten(tree_struct, best_params)

    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
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

    N = HomothetyOperator(jnp.ones(1), _in_structure=masked_d.structure)
    solver = optax.lbfgs()

    inverser_options = {
        "solver": lx.CG(rtol=1e-6, atol=1e-6, max_steps=1000),
        "solver_throw": False,
    }

    @partial(jax.jit, static_argnums=(5))
    def compute_minimum_variance(
        T_d_patches, B_d_patches, B_s_patches, planck_mask, indices, max_patches=25
    ):
        temp_dust_patch_indices = None
        beta_dust_patch_indices = get_clusters(
            planck_mask,
            indices,
            B_d_patches,
            jax.random.PRNGKey(0),
            max_centroids=max_patches,
        )
        beta_pl_patch_indices = None

        params = {
            "beta_dust": jnp.full((max_patches,), 1.54),
            "temp_dust": jnp.full((1,), 20.0),
            "beta_pl": jnp.full((1,), (-3.0)),
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
                max_iter=100,
                tol=1e-8,
                verbose=True,
                log_interval=0.05,
                nu=nu,
                N=N,
                d=masked_d,
                patch_indices=masked_clusters,
            )

        value = spectral_cmb_variance_fn(
            final_params, nu=nu, d=masked_d, N=N, patch_indices=masked_clusters
        )

        return {
            "value": value,
            "beta_dust": final_params["beta_dust"],
            "temp_dust": final_params["temp_dust"],
            "beta_pl": final_params["beta_pl"],
        }

    max_centroids = 200
    search_space = {
        "T_d_patches": jnp.array([1]),
        "B_d_patches": jnp.arange(10, 201, 10),
        "B_s_patches": jnp.array([1]),
    }

    @jax.jit
    def objective_function(T_d_patches, B_d_patches, B_s_patches):
        return compute_minimum_variance(
            T_d_patches,
            B_d_patches,
            B_s_patches,
            mask,
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
        result_dir="validation_model",
    )

    grid_search.run()

    results = grid_search.stack_results(result_folder="validation_model")
    # Save results
    np.savez("results.npz", **results)
    np.savez("best_params.npz", **best_params)


if __name__ == "__main__":
    main()
