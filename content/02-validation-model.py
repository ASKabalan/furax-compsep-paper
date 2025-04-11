import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import operator
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from furax import HomothetyOperator
from furax._instruments.sky import FGBusterInstrument
from furax.comp_sep import (
    negative_log_likelihood,
    sky_signal,
    spectral_cmb_variance,
)
from furax.obs.landscapes import HealpixLandscape
from jax_grid_search import DistributedGridSearch, ProgressBar, optimize
from jax_healpy import get_clusters, get_cutout_from_mask
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, simulate_D_from_params
from plotting import plot_cmb_nll_vs_B_d_patches, plot_healpix_projection


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
        "-p",
        "--plot",
        action="store_true",
        help="Plot the results",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020",
        choices=MASK_CHOICES,
        help="Mask to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_folder = f"validation_model_{args.mask}"
    if args.plot:
        assert os.path.exists(out_folder), "Validation model not found, please run the model first"

        results = np.load(f"{out_folder}/results.npz", allow_pickle=True)
        best_params = np.load(f"{out_folder}/best_params.npz", allow_pickle=True)
        mask = np.load(f"{out_folder}/mask.npy", allow_pickle=True)
        best_params = dict(best_params)
        results = dict(results)
        plot_cmb_nll_vs_B_d_patches(results, best_params, out_folder)
        plot_healpix_projection(mask, args.nside, results, best_params, out_folder)
        return

    nside = args.nside
    dust_nu0 = 150.0
    synchrotron_nu0 = 20.0
    max_centroids = 300
    patch_indices = {
        "temp_dust_patches": None,
        "beta_dust_patches": 160,
        "beta_pl_patches": None,
    }
    get_count = lambda c: c if c is not None else 1  # noqa E731
    params_count = {
        "beta_dust": get_count(patch_indices["beta_dust_patches"]),
        "temp_dust": get_count(patch_indices["temp_dust_patches"]),
        "beta_pl": get_count(patch_indices["beta_pl_patches"]),
    }

    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    patch_indices = jax.tree.map(
        lambda c: get_clusters(mask, indices, c, jax.random.key(0), max_centroids=max_centroids),
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

    nu = FGBusterInstrument.default_instrument().frequency
    land_scape = HealpixLandscape(nside=nside, stokes="QU")

    sky = {
        "cmb": land_scape.normal(jax.random.key(0)),
        "dust": land_scape.normal(jax.random.key(1)),
        "synchrotron": land_scape.normal(jax.random.key(2)),
    }
    masked_sky = get_cutout_from_mask(sky, indices)

    params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, params_count)
    params_flat, tree_struct = jax.tree.flatten(params)

    params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].idx), x.shape) * 0.2,
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
    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)

    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    best_nll = negative_log_likelihood_fn(
        best_params,
        nu=nu,
        d=masked_d,
        N=HomothetyOperator(jnp.ones(1), _in_structure=masked_d.structure),
        patch_indices=masked_clusters,
    )
    best_cmb_var = spectral_cmb_variance_fn(
        best_params,
        nu=nu,
        d=masked_d,
        N=HomothetyOperator(jnp.ones(1), _in_structure=masked_d.structure),
        patch_indices=masked_clusters,
    )

    N = HomothetyOperator(jnp.ones(1), _in_structure=masked_d.structure)

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
        guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int32), guess_clusters)

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        solver = optax.lbfgs()
        final_params, final_state = optimize(
            guess_params,
            negative_log_likelihood_fn,
            solver,
            max_iter=1000,
            tol=1e-10,
            lower_bound=lower_bound_tree,
            upper_bound=upper_bound_tree,
            nu=nu,
            N=N,
            d=masked_d,
            patch_indices=guess_clusters,
        )

        s = sky_signal_fn(final_params, nu=nu, d=masked_d, N=N, patch_indices=guess_clusters)
        cmb = s["cmb"]
        cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

        cmb_np = jnp.stack([cmb.q, cmb.u])

        nll = negative_log_likelihood_fn(
            final_params, nu=nu, d=masked_d, N=N, patch_indices=guess_clusters
        )

        return {
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

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    # Put the good values for the grid search
    if os.path.exists(out_folder):
        old_results = DistributedGridSearch.stack_results(result_folder=out_folder)
    else:
        old_results = None

    with ProgressBar(*progress_columns) as p:

        @jax.jit
        def objective_function(T_d_patches, B_d_patches, B_s_patches):
            return compute_minimum_variance(
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
            batch_size=4,
            progress_bar=True,
            log_every=0.1,
            result_dir=out_folder,
            old_results=old_results,
        )

        grid_search.run()

    results = grid_search.stack_results(result_folder=out_folder)

    # Save results
    cmb_map = np.stack([masked_sky["cmb"].q, masked_sky["cmb"].u])
    best_params["I_CMB"] = cmb_map
    best_params["NLL"] = best_nll
    best_params["value"] = best_cmb_var
    best_params["B_d_patches"] = params_count["beta_dust"]
    best_params["T_d_patches"] = params_count["temp_dust"]
    best_params["B_s_patches"] = params_count["beta_pl"]
    best_params["beta_dust_patches"] = masked_clusters["beta_dust_patches"]
    best_params["temp_dust_patches"] = masked_clusters["temp_dust_patches"]
    best_params["beta_pl_patches"] = masked_clusters["beta_pl_patches"]
    np.savez(f"{out_folder}/results.npz", **results)
    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", mask)


if __name__ == "__main__":
    main()
