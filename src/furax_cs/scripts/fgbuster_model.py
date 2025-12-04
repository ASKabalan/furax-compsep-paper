#!/usr/bin/env python3
"""
FGBuster Component Separation for CMB Polarization Analysis

This script implements adaptive K-means clustering for optimizing parametric
component separation in CMB polarization analysis using FGBuster.
The method uses spherical K-means to partition the sky based on spatial coordinates,
then performs component separation using FGBuster's adaptive_comp_sep function.

This is the NumPy/FGBuster equivalent of kmeans_model.py which uses JAX/Furax.

Usage:
    python fgbuster_model.py -n 64 -pc 100 5 1 -tag c1d1s1 -m GAL020 -i LiteBIRD

Parameters:
    -pc: Number of clusters for [dust_beta, dust_temp, sync_beta] parameters
    -tag: Sky simulation configuration tag
    -m: Galactic mask (GAL020, GAL040, GAL060)
    -i: Instrument specification

Output:
    Results saved to results/fgbuster_{config}_{instrument}_{mask}_{samples}/
    - best_params.npz: Optimized spectral parameters per cluster
    - results.npz: Full clustering and component separation results
    - mask.npy: Sky mask used for analysis

Note:
    FGBuster operates with NumPy, not JAX. Multiple noise simulations are not
    supported in this implementation (use kmeans_model.py for Monte Carlo analysis).

Author: FURAX Team
"""

import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
from time import perf_counter

import healpy as hp
import jax
import jax.numpy as jnp
import jax.random
import numpy as np

try:
    from fgbuster import (
        CMB,
        Dust,
        Synchrotron,
        adaptive_comp_sep,
    )
    from fgbuster import (
        get_instrument as get_fgbuster_instrument,
    )
except ImportError:
    raise ImportError(
        "FGBuster is required for this baseline comparison script. Install with:\n"
        "  pip install fgbuster\n"
        "or\n"
        "  pip install git+https://github.com/fgbuster/fgbuster.git"
    )

from furax.obs.stokes import Stokes
from jax_healpy.clustering import (
    find_kmeans_clusters,
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

jax.config.update("jax_enable_x64", True)


def parse_args():
    """
    Parse command-line arguments for FGBuster component separation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - nside: HEALPix resolution parameter
        - noise_sim: Number of noise simulations (must be 1 for FGBuster)
        - noise_ratio: Noise level relative to signal
        - tag: Sky simulation configuration identifier
        - mask: Galactic mask choice (GAL020, GAL040, GAL060)
        - instrument: Instrument configuration (LiteBIRD, Planck)
        - patch_count: Target cluster counts for [dust_beta, dust_temp, sync_beta]
        - best_only: Flag to only compute optimal configuration
    """
    parser = argparse.ArgumentParser(
        description="FGBuster Component Separation for CMB Polarization Analysis"
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
        help="Number of noise simulations (must be 1 for FGBuster implementation)",
    )
    parser.add_argument(
        "-nr",
        "--noise-ratio",
        type=float,
        default=0.2,
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
        nargs=3,
        default=[1000, 10, 10],
        help=(
            "List of three target patch counts for beta_dust, temp_dust, and beta_pl. "
            "Example: --patch-count 1000 10 10"
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
        help="Maximum number of optimization iterations for TNC solver",
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
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    return parser.parse_args()


def run_fgbuster_comp_sep(
    freq_maps,
    patch_ids_fg,
    components,
    instrument,
    max_iter=1000,
    tol=1e-6,
):
    """
    Run FGBuster adaptive component separation.

    Parameters
    ----------
    freq_maps : ndarray
        Frequency maps with shape (n_freq, n_stokes, n_pix) or (n_freq, 2, n_pix) for QU
    patch_ids_fg : list of ndarray
        List of patch indices for [temp_dust, beta_dust, beta_pl]
    dust_nu0 : float
        Dust reference frequency in GHz
    synchrotron_nu0 : float
        Synchrotron reference frequency in GHz
    instrument : dict
        FGBuster instrument specification
    max_iter : int
        Maximum optimization iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    result : object
        FGBuster result object containing:
        - x: optimized parameters
        - s: separated components (CMB, dust, synchrotron)
        - fun: final objective function value
    """

    # FGBuster bounds: [beta_dust, temp_dust, beta_pl]
    bounds = [(0.0, 5.0), (10.0, 40.0), (-6.0, 0.0)]

    options = {
        "disp": False,
        "gtol": tol,
        "eps": tol,
        "maxiter": max_iter,
        "tol": tol,
    }
    method = "L-BFGS-B"

    # Convert to numpy arrays for FGBuster
    freq_maps_np = np.asarray(freq_maps)
    patch_ids_np = [np.asarray(p) for p in patch_ids_fg]

    result = adaptive_comp_sep(
        components,
        instrument,
        freq_maps_np,
        patch_ids_np,
        bounds=bounds,
        options=options,
        method=method,
        tol=tol,
    )

    return result


def main():
    """
    Main execution function for FGBuster component separation.

    Implements the adaptive clustering algorithm that partitions the sky into
    regions with spatially-varying spectral parameters using FGBuster's
    adaptive_comp_sep function.

    Algorithm Steps:
    1. Initialize sky masks and clustering parameters
    2. Load CMB and foreground simulations
    3. Perform spherical K-means clustering on sky coordinates
    4. Run FGBuster adaptive component separation
    5. Evaluate CMB reconstruction variance
    6. Save clustering configuration and parameters
    """
    # Step 1: Parse arguments and validate
    args = parse_args()

    # FGBuster doesn't support multiple noise simulations in this implementation
    if args.noise_sim > 1:
        raise NotImplementedError(
            f"Multiple noise simulations (noise_sim={args.noise_sim}) are not supported "
            "in the FGBuster implementation. Use noise_sim=1 or use kmeans_model.py "
            "for Monte Carlo analysis with JAX/Furax."
        )

    # Setup output directory
    patches = f"BD{args.patch_count[0]}_TD{args.patch_count[1]}_BS{args.patch_count[2]}"
    out_folder = f"{args.output}/fgbuster_{args.tag}_{patches}_{args.instrument}_{sanitize_mask_name(args.mask)}_{int(args.noise_ratio * 100)}"
    os.makedirs(out_folder, exist_ok=True)

    # Step 2: Initialize physical and computational parameters
    nside = args.nside
    dust_nu0 = 150.0  # Dust reference frequency (GHz) - FGBuster convention
    synchrotron_nu0 = 20.0  # Synchrotron reference frequency (GHz)

    # Step 3: Load galactic mask and extract valid pixel indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    # Step 4: Determine maximum cluster counts (limited by available pixels)
    B_dust_patches = min(args.patch_count[0], indices.size)
    T_dust_patches = min(args.patch_count[1], indices.size)
    B_synchrotron_patches = min(args.patch_count[2], indices.size)

    # Step 5: Load frequency maps and component maps
    print(f"Loading data for nside={nside}, tag={args.tag}, instrument={args.instrument}")
    _, freqmaps = load_from_cache(nside, noise=False, instrument_name=args.instrument, sky=args.tag)
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)

    # Create Stokes objects for data handling
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])

    # Apply mask cutouts
    masked_d = jax.tree.map(lambda x: x.at[..., indices].set(hp.UNSEEN), d)
    masked_fg = jax.tree.map(lambda x: x.at[..., indices].set(hp.UNSEEN), fg_stokes)
    masked_cmb = jax.tree.map(lambda x: x.at[indices].set(hp.UNSEEN), cmb_map_stokes)

    jax.tree.map_with_path(lambda p, x: print(f"D {p}: shape {x.shape}"), d)
    jax.tree.map_with_path(lambda p, x: print(f"FG {p}: shape {x.shape}"), fg_stokes)
    jax.tree.map_with_path(lambda p, x: print(f"CMB {p}: shape {x.shape}"), cmb_map_stokes)
    jax.tree.map_with_path(lambda p, x: print(f"Masked D {p}: shape {x.shape}"), masked_d)
    jax.tree.map_with_path(lambda p, x: print(f"Masked FG {p}: shape {x.shape}"), masked_fg)
    jax.tree.map_with_path(lambda p, x: print(f"Masked CMB {p}: shape {x.shape}"), masked_cmb)
    # to numpy for FGBuster
    masked_d = jax.tree.map(np.asarray, masked_d)
    masked_fg = jax.tree.map(np.asarray, masked_fg)
    masked_cmb = jax.tree.map(np.asarray, masked_cmb)

    # Step 6: Perform K-means clustering for patch indices
    print(
        f"Computing K-means clusters: T_dust={T_dust_patches}, B_dust={B_dust_patches}, B_sync={B_synchrotron_patches}"
    )

    max_patches = {
        "temp_dust_patches": T_dust_patches,
        "beta_dust_patches": B_dust_patches,
        "beta_pl_patches": B_synchrotron_patches,
    }
    patch_indices = jax.tree.map(
        lambda c, mp: find_kmeans_clusters(
            mask, indices, c, jax.random.key(0), max_centroids=mp, initial_sample_size=1
        ),
        max_patches,
        max_patches,
    )
    guess_clusters = jax.tree.map(
        lambda x: np.where(x == hp.UNSEEN, x.max(), x).astype(int), patch_indices
    )
    # Normalize the cluster to make indexing more logical
    guess_clusters = jax.tree.map(
        lambda g, c, mp: normalize_by_first_occurrence(g, c, mp).astype(jnp.int64),
        guess_clusters,
        max_patches,
        max_patches,
    )
    guess_clusters = jax.tree.map(lambda x: jnp.where(mask == 1, x, x.max()), guess_clusters)
    patch_ids_fg = [
        np.asarray(guess_clusters["beta_dust_patches"]),
        np.asarray(guess_clusters["temp_dust_patches"]),
        np.asarray(guess_clusters["beta_pl_patches"]),
    ]

    # Step 7: Prepare frequency maps for FGBuster
    # FGBuster expects shape (n_freq, n_stokes, n_pix) with Q, U stokes
    freq_maps_fg = jnp.stack([masked_d.q, masked_d.u], axis=1)

    # Get FGBuster instrument
    instrument = get_fgbuster_instrument(args.instrument)

    # Step 8: Run FGBuster component separation
    print(f"Running FGBuster adaptive_comp_sep with max_iter={args.max_iter}")
    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
    components[1]._set_default_of_free_symbols(
        beta_d=args.starting_params[0], temp=args.starting_params[1]
    )
    components[2]._set_default_of_free_symbols(beta_pl=args.starting_params[2])

    start_time = perf_counter()
    result = run_fgbuster_comp_sep(
        freq_maps_fg,
        patch_ids_fg,
        components,
        instrument,
        max_iter=args.max_iter,
    )
    end_time = perf_counter()
    print(f"Component separation took {end_time - start_time:.2f} seconds")

    # Step 9: Extract results
    cmb_q, cmb_u = result.s[0]  # CMB is the first component
    cmb_var = np.var(cmb_q) + np.var(cmb_u)

    print(f"Component separation complete. CMB variance: {cmb_var:.6e}")
    print(f"Final objective function value: {result.fun}")

    # Step 10: Prepare results for saving

    # Prepare results dictionary
    cmb_np = np.stack([cmb_q, cmb_u])

    results = {
        "value": np.array([cmb_var])[np.newaxis, ...],  # Add axes to match kmeans_model format
        "CMB_O": cmb_np[np.newaxis, np.newaxis, ...],
        "NLL": np.array([result.fun])[np.newaxis, ...],
        "beta_dust": result.x[0][np.newaxis, np.newaxis, ...],
        "temp_dust": result.x[1][np.newaxis, np.newaxis, ...],
        "beta_pl": result.x[2][np.newaxis, np.newaxis, ...],
        "beta_dust_patches": np.asarray(guess_clusters["beta_dust_patches"])[np.newaxis, ...],
        "temp_dust_patches": np.asarray(guess_clusters["temp_dust_patches"])[np.newaxis, ...],
        "beta_pl_patches": np.asarray(guess_clusters["beta_pl_patches"])[np.newaxis, ...],
    }

    if not args.best_only:
        np.savez(f"{out_folder}/results.npz", **results)

    # Save best parameters and auxiliary data
    best_params = {}
    cmb_map_arr = np.stack([np.asarray(masked_cmb.q), np.asarray(masked_cmb.u)], axis=0)
    fg_map_arr = np.stack([np.asarray(masked_fg.q), np.asarray(masked_fg.u)], axis=1)
    d_map_arr = np.stack([np.asarray(masked_d.q), np.asarray(masked_d.u)], axis=1)

    best_params["I_CMB"] = cmb_map_arr
    best_params["I_D"] = d_map_arr
    best_params["I_D_NOCMB"] = fg_map_arr

    np.savez(f"{out_folder}/best_params.npz", **best_params)
    np.save(f"{out_folder}/mask.npy", np.asarray(mask))

    print(f"Run complete. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
