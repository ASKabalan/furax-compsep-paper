#!/usr/bin/env python3
"""
K-Means Clustering for Adaptive CMB Component Separation

This script implements the main contribution of the FURAX framework: adaptive K-means
clustering for optimizing parametric component separation in CMB polarization analysis.
The method uses spherical K-means to partition the sky based on spatial coordinates,
then performs variance-based model selection to determine optimal cluster configurations.

Key Innovation:
    - Spherical K-means clustering using RA/Dec coordinates with 3D Cartesian averaging
    - Variance-based selection to minimize CMB reconstruction contamination
    - Distributed grid search across clustering configurations
    - Adaptive sky partitioning for spatially-varying spectral parameters

Usage:
    python 08-KMeans-model.py -n 64 -pc 100 5 1 -tag c1d1s1 -m GAL020 -i LiteBIRD

Parameters:
    -pc: Number of clusters for [dust_temp, dust_beta, sync_beta] parameters
    -tag: Sky simulation configuration tag
    -m: Galactic mask (GAL020, GAL040, GAL060)
    -i: Instrument specification

Output:
    Results saved to results/kmeans_{config}_{instrument}_{mask}_{samples}/
    - best_params.npz: Optimized spectral parameters per cluster
    - results.npz: Full clustering and component separation results
    - mask.npy: Sky mask used for analysis

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"

import argparse
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
import operator
from time import perf_counter

import jax.numpy as jnp
import jax.random
import lineax as lx
import numpy as np
from furax import Config
from furax._instruments.sky import (
    get_noise_sigma_from_instrument,
)
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_healpy.clustering import (
    find_kmeans_clusters,
    get_cutout_from_mask,
    normalize_by_first_occurrence,
)

from furax_cs.data.generate_maps import (
    MASK_CHOICES,
    get_mask,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    sanitize_mask_name,
)
from furax_cs.data.instruments import get_instrument
from furax_cs.logging_utils import info, success
from furax_cs.optim import minimize

jax.config.update("jax_enable_x64", True)


def parse_args():
    """
    Parse command-line arguments for K-means clustering component separation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - nside: HEALPix resolution parameter
        - noise_sim: Number of noise simulations for MC analysis
        - noise_ratio: Noise level relative to signal
        - tag: Sky simulation configuration identifier
        - mask: Galactic mask choice (GAL020, GAL040, GAL060)
        - instrument: Instrument configuration (LiteBIRD, Planck)
        - patch_count: Target cluster counts for [dust_beta, dust_temp, sync_beta]
        - best_only: Flag to only compute optimal configuration
    """
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Adaptive CMB Component Separation"
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="HEALPix nside parameter determining map resolution (nside=64 → ~55 arcmin pixels)",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=1,
        help="Number of Monte Carlo noise realizations for statistical analysis",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.0,
        help="Noise level as fraction of signal RMS (0.2 = 20%% noise)",
    )
    parser.add_argument(
        "-tag",
        "--tag",
        type=str,
        default="c1d1s1",
        help="Sky simulation tag: c(CMB)d(dust)s(synchrotron) with 0/1 for off/on",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="GAL020_U",
        help=f"Galactic mask: GAL020/040/060 (20%%/40%%/60%% sky coverage), _U/_L for upper/lower. "
        f"Available masks: {MASK_CHOICES}. "
        "Combine with + (union) or - (subtract), e.g., GAL020+GAL040 or ALL-GALACTIC",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument configuration with frequency bands and noise characteristics",
    )
    parser.add_argument(
        "-pc",
        "--patch-count",
        type=int,
        nargs=3,  # Expecting exactly three values
        default=[10000, 500, 500],  # Example target patch counts for beta_dust, temp_dust, beta_pl
        help=(
            "List of three target patch counts for beta_dust, temp_dust, and beta_pl. "
            "Example: --patch-count 10000 500 500"
        ),
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
        "-sp",
        "--starting-params",
        type=float,
        nargs=3,
        default=[1.54, 20.0, -3.0],
        help="Starting parameters for [beta_dust, temp_dust, beta_pl]",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="optax_lbfgs_zoom",
        help="Solver for optimization. Options: optax_lbfgs_zoom, optax_lbfgs_backtrack, "
        "optimistix_bfgs_wolfe, optimistix_lbfgs_wolfe, optimistix_ncg_hs_wolfe, "
        "scipy_tnc, zoom (alias), backtrack (alias), adam",
    )
    parser.add_argument(
        "-cond",
        "--cond",
        action="store_true",
        help="Enable conditioning (pre and post) for L-BFGS solver",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    return parser.parse_args()


def main():
    """
    Main execution function for K-means clustering component separation.

    Implements the adaptive clustering algorithm that partitions the sky into
    regions with spatially-varying spectral parameters. Uses variance-based
    model selection to determine optimal cluster configurations.

    Algorithm Steps:
    1. Initialize sky masks and clustering parameters
    2. Load CMB and foreground simulations
    3. Perform spherical K-means clustering on sky coordinates
    4. Optimize spectral parameters within each cluster
    5. Evaluate CMB reconstruction variance for model selection
    6. Save optimal clustering configuration and parameters
    """
    # Step 1: Parse arguments and setup output directory
    args = parse_args()

    patches = f"BD{args.patch_count[0]}_TD{args.patch_count[1]}_BS{args.patch_count[2]}_SP{args.starting_params[0]}_{args.starting_params[1]}_{args.starting_params[2]}"
    config = f"{args.solver}_cond{args.cond}_noise{int(args.noise_ratio * 100)}"
    out_folder = f"{args.output}/kmeans_{args.tag}_{patches}_{args.instrument}_{sanitize_mask_name(args.mask)}_{config}"

    # Step 2: Initialize physical and computational parameters
    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0  # Dust reference frequency (GHz)
    synchrotron_nu0 = 20.0  # Synchrotron reference frequency (GHz)

    # Step 3: Load galactic mask and extract valid pixel indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)  # Get indices of unmasked pixels

    # Step 4: Determine maximum cluster counts (limited by available pixels)
    B_dust_patches = min(args.patch_count[0], indices.size)  # Dust spectral index clusters
    T_dust_patches = min(args.patch_count[1], indices.size)  # Dust temperature clusters
    B_synchrotron_patches = min(args.patch_count[2], indices.size)  # Synchrotron index clusters

    base_params = {
        "beta_dust": args.starting_params[0],
        "temp_dust": args.starting_params[1],
        "beta_pl": args.starting_params[2],
    }
    lower_bound = {
        "beta_dust": 0.5,
        "temp_dust": 10.0,
        "beta_pl": -7.0,
    }
    upper_bound = {
        "beta_dust": 3.0,
        "temp_dust": 40.0,
        "beta_pl": -0.5,
    }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    sky_signal_fn = partial(sky_signal, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    negative_log_likelihood_fn = partial(
        negative_log_likelihood,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        analytical_gradient=True,
    )

    _, freqmaps = load_from_cache(nside, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)
    masked_fg = get_cutout_from_mask(fg_stokes, indices, axis=1)
    masked_cmb = get_cutout_from_mask(cmb_map_stokes, indices)

    max_count = {
        "beta_dust": B_dust_patches,
        "temp_dust": T_dust_patches,
        "beta_pl": B_synchrotron_patches,
    }
    max_patches = {
        "temp_dust_patches": max_count["temp_dust"],
        "beta_dust_patches": max_count["beta_dust"],
        "beta_pl_patches": max_count["beta_pl"],
    }

    @jax.jit
    def compute_minimum_variance(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        planck_mask,
        indices,
    ):
        n_regions = {
            "temp_dust_patches": T_d_patches,
            "beta_dust_patches": B_d_patches,
            "beta_pl_patches": B_s_patches,
        }

        patch_indices = jax.tree.map(
            lambda c, mp: find_kmeans_clusters(
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
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

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

            final_params, final_state = minimize(
                fn=negative_log_likelihood_fn,
                init_params=guess_params,
                solver_name=args.solver,
                max_iter=args.max_iter,
                atol=1e-15,
                rtol=1e-10,
                lower_bound=lower_bound_tree,
                upper_bound=upper_bound_tree,
                precondition=args.cond,
                nu=nu,
                N=N,
                d=noised_d,
                patch_indices=guess_clusters,
            )
            s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=guess_clusters)
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

        results_list = [single_run(i) for i in range(nb_noise_sim)]
        results = jax.tree.map(lambda *xs: jnp.stack(xs), *results_list)
        results["beta_dust_patches"] = guess_clusters["beta_dust_patches"]
        results["temp_dust_patches"] = guess_clusters["temp_dust_patches"]
        results["beta_pl_patches"] = guess_clusters["beta_pl_patches"]
        return results

    with Config(solver=lx.CG(atol=1e-10, rtol=1e-6, max_steps=1000)), jax.disable_jit(False):

        @jax.jit
        def objective_function(T_d_patches, B_d_patches, B_s_patches):
            return compute_minimum_variance(
                T_d_patches,
                B_d_patches,
                B_s_patches,
                mask,
                indices,
            )

        if not args.best_only:
            start_time = perf_counter()
            results = objective_function(T_dust_patches, B_dust_patches, B_synchrotron_patches)
            jax.tree.map(lambda x: x.block_until_ready(), results)
            end_time = perf_counter()
            min_bd, max_bd = results["beta_dust"].min(), results["beta_dust"].max()
            min_td, max_td = results["temp_dust"].min(), results["temp_dust"].max()
            min_bs, max_bs = results["beta_pl"].min(), results["beta_pl"].max()
            info(f"min beta_dust: {min_bd} max beta_dust: {max_bd}")
            info(f"min temp_dust: {min_td} max temp_dust: {max_td}")
            info(f"min beta_pl: {min_bs} max beta_pl: {max_bs}")
            info(f"Objective function evaluation took {end_time - start_time:.2f} seconds")
            # Add a new axis to the results so it matches the shape of grid search results
            os.makedirs(out_folder, exist_ok=True)
            results = jax.tree.map(lambda x: x[np.newaxis, ...], results)
            np.savez(f"{out_folder}/results.npz", **results)

    os.makedirs(out_folder, exist_ok=True)
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
    success(f"Run complete. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
