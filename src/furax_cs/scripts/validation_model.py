import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import operator
from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from furax import HomothetyOperator
from furax._instruments.sky import FGBusterInstrument
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
    spectral_cmb_variance,
)
from furax.obs.landscapes import HealpixLandscape
from jax_grid_search import DistributedGridSearch
from jax_healpy.clustering import (
    find_kmeans_clusters,
    get_cutout_from_mask,
    get_fullmap_from_cutout,
)

from furax_cs.data.generate_maps import (
    MASK_CHOICES,
    get_mask,
    sanitize_mask_name,
    simulate_D_from_params,
)
from furax_cs.logging_utils import success
from furax_cs.optim import minimize

jax.config.update("jax_enable_x64", True)


def plot_cmb_nll_vs_B_d_patches(results, best_params, out_folder):
    """Plot CMB variance and NLL vs number of dust index patches.

    Parameters
    ----------
    results : dict
        Grid search results dictionary.
    best_params : dict
        Best parameter configuration.
    out_folder : str
        Output directory for saving plots.
    """
    sns.set_context("paper")
    # Extract values from the grid search results
    B_d_patches = results["B_d_patches"]  # dust patch count (x-axis)
    cmb_variance = results["value"]  # CMB variance values from grid search
    nll = results["NLL"]  # Negative log-likelihood values

    # Create subplots: one for CMB variance, one for NLL
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Handle case where values have multiple realizations (shape: n_configs, n_realizations)
    if cmb_variance.ndim > 1 and cmb_variance.shape[1] > 1:
        # Compute mean and std across noise realizations
        variance_mean = np.mean(cmb_variance, axis=1)
        variance_std = np.std(cmb_variance, axis=1)
        nll_mean = np.mean(nll, axis=1)
        nll_std = np.std(nll, axis=1)

        # Plot with error bars
        axs[0].errorbar(
            B_d_patches,
            variance_mean,
            yerr=variance_std,
            fmt="o-",
            capsize=5,
            capthick=2,
            color="blue",
            label="Grid Search",
        )
        axs[1].errorbar(
            B_d_patches,
            nll_mean,
            yerr=nll_std,
            fmt="o-",
            capsize=5,
            capthick=2,
            color="green",
            label="Grid Search",
        )
    else:
        # Single realization case - use scatter plot
        axs[0].scatter(B_d_patches, cmb_variance, color="blue", label="Grid Search")
        axs[1].scatter(B_d_patches, nll, color="green", label="Grid Search")

    # Add best parameter lines
    axs[0].axhline(y=best_params["value"], color="red", linestyle="--", label="Best CMB Variance")
    axs[0].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label=r"Best $K_{\beta_d}$",
    )
    axs[0].set_xlabel(r"$K_{\beta_d}$")  # Updated to K-notation
    axs[0].set_ylabel(r"CMB Variance ($\mu$K²)")  # Added units
    axs[0].set_title(r"CMB Variance vs. $K_{\beta_d}$")  # Updated title
    axs[0].legend()

    # Plot NLL vs. B_d_patches
    axs[1].axhline(y=best_params["NLL"], color="red", linestyle="--", label="Best NLL")
    axs[1].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label=r"Best $K_{\beta_d}$",
    )
    axs[1].set_xlabel(r"$K_{\beta_d}$")  # Updated to K-notation
    axs[1].set_ylabel("Negative Log Likelihood")
    axs[1].set_title(r"NLL vs. $K_{\beta_d}$")  # Updated title
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/cmb_nll_vs_B_d_patches.png",
        dpi=1200,
        transparent=True,
    )


def plot_healpix_projection(mask, nside, results, best_params, out_folder):
    """Project best beta_dust parameters onto full HEALPix sky.

    Parameters
    ----------
    mask : ndarray
        Boolean sky mask.
    nside : int
        HEALPix resolution parameter.
    results : dict
        Grid search results.
    best_params : dict
        Best parameter configuration.
    out_folder : str
        Output directory path.
    """
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)
    best_run = jax.tree.map(lambda x: x[0], results)
    patches = best_run["beta_dust_patches"]
    best_spectral_params = best_params["beta_dust"][patches]
    result_spectral_params = best_run["beta_dust"][patches]

    best_healpix_map = get_fullmap_from_cutout(best_spectral_params, indices, nside)
    result_healpix_map = get_fullmap_from_cutout(result_spectral_params, indices, nside)

    # Plot the best and result maps
    plt.figure(figsize=(15, 5))
    hp.mollview(best_healpix_map, title="Best Beta Dust map", sub=(1, 2, 1), bgcolor=(0.0,) * 4)
    hp.mollview(
        result_healpix_map,
        title="Result Beta Dust map",
        sub=(1, 2, 2),
        bgcolor=(0.0,) * 4,
    )
    plt.savefig(
        f"{out_folder}/best_result_healpix_projection.png",
        dpi=1200,
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
        help=f"Mask to use. Available masks: {MASK_CHOICES}. "
        "Combine with + (union) or - (subtract), e.g., GAL020+GAL040 or ALL-GALACTIC",
    )
    parser.add_argument(
        "-b",
        "--best-only",
        action="store_true",
        help="Only generate best results",
    )
    parser.add_argument(
        "-mi",
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations for L-BFGS solver",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="optax_lbfgs_zoom",
        help="Solver for optimization. Options: optax_lbfgs_zoom, optax_lbfgs_backtrack, "
        "optimistix_bfgs, optimistix_lbfgs, scipy_tnc, adam",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_folder = f"validation_model_{sanitize_mask_name(args.mask)}"
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
        lambda c: find_kmeans_clusters(
            mask, indices, c, jax.random.key(0), max_centroids=max_centroids
        ),
        patch_indices,
    )
    masked_clusters = get_cutout_from_mask(patch_indices, indices)
    masked_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), masked_clusters)

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }
    lower_bound = {
        "beta_dust": 0.5,
        "temp_dust": 6.0,
        "beta_pl": -7.0,
    }
    upper_bound = {
        "beta_dust": 5.0,
        "temp_dust": 40.0,
        "beta_pl": -0.5,
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

    masked_d, _ = simulate_D_from_params(
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
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
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

    @partial(jax.jit, static_argnums=(5,))
    def compute_minimum_variance(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        planck_mask,
        indices,
        max_patches=25,
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
            lambda c: find_kmeans_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=max_centroids
            ),
            patch_indices,
        )
        guess_clusters = get_cutout_from_mask(patch_indices, indices)
        guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        final_params, final_state = minimize(
            fn=negative_log_likelihood_fn,
            init_params=guess_params,
            solver_name=args.solver,
            max_iter=args.max_iter,
            rtol=1e-10,
            atol=1e-10,
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

    # Put the good values for the grid search
    if os.path.exists(out_folder):
        old_results = DistributedGridSearch.stack_results(result_folder=out_folder)
    else:
        old_results = None

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

    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=4,
        progress_bar=True,
        log_every=0.1,
        result_dir=out_folder,
        old_results=old_results,
    )

    if not args.best_only:
        grid_search.run()

    if not args.best_only:
        results = grid_search.stack_results(result_folder=out_folder)
        np.savez(f"{out_folder}/results.npz", **results)

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
    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", mask)
    success(f"Run complete. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
