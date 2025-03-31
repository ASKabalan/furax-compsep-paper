import os

os.environ["EQX_ON_ERROR"] = "nan"

import argparse
import os
import sys
from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from furax._instruments.sky import get_noise_sigma_from_instrument
from furax.comp_sep import negative_log_likelihood, spectral_cmb_variance
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_grid_search import ProgressBar, optimize
from jax_healpy import get_cutout_from_mask
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Make sure these modules are available in your PYTHONPATH
sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, load_cmb_map, load_from_cache
from instruments import get_instrument


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Benchmark FGBuster and Furax Component Separation Methods (single run with ud_grade)
        """
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="The nside of the input map",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=1,
        help="Number of noise simulations (single run uses 1)",
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
        choices=["LiteBIRD", "Planck"],
        help="Instrument to use",
    )
    parser.add_argument(
        "-ud",
        "--target-ud-grade",
        type=float,
        nargs=3,  # Expecting exactly three values
        default=[64, 32, 16],  # Example target nsides for beta_dust, temp_dust, beta_pl
        help=(
            "List of three target nside values (for ud_grade downgrading) corresponding to "
            "beta_dust, temp_dust, beta_pl respectively"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Define the output folder and create it if necessary
    out_folder = f"PTEP_{args.tag}_{args.instrument}_{args.mask}"
    os.makedirs(out_folder, exist_ok=True)

    # Set up parameters
    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    # Get the mask and its indices
    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    # Load frequency maps and extract the Stokes Q/U maps (for example)
    _, freqmaps = load_from_cache(nside, noise=False, instrument_name=args.instrument, sky=args.tag)
    _, cmb_map = load_cmb_map(nside, noise=False, instrument_name=args.instrument, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)

    # Instead of clustering, use hp.ud_grade to downgrade to lower resolutions.
    # We interpret the three ud values as the target nsides for each parameter.
    # (Make sure these values are integers and valid Healpy nsides.)
    ud_beta_d = int(args.target_ud_grade[0])
    ud_temp_d = int(args.target_ud_grade[1])
    ud_beta_pl = int(args.target_ud_grade[2])

    # Create a dummy full-sky map (of ones) to generate the downgraded patch indices.
    npix = nside**2 * 12
    ipix = np.arange(npix)

    def ud_grade(ipix, nside_out):
        if nside_out == 0:
            return np.zeros_like(ipix)
        else:
            return hp.ud_grade(ipix, nside_out=nside_out)

    ud_beta_d_map = ud_grade(ipix, ud_beta_d)
    ud_temp_d_map = ud_grade(ipix, ud_temp_d)
    ud_beta_pl_map = ud_grade(ipix, ud_beta_pl)

    # These downgraded maps serve as our patch indices.
    patch_indices = {
        "beta_dust_patches": ud_beta_d_map,
        "temp_dust_patches": ud_temp_d_map,
        "beta_pl_patches": ud_beta_pl_map,
    }
    max_count = {
        "beta_dust": np.unique(ud_beta_d_map).size,
        "temp_dust": np.unique(ud_temp_d_map).size,
        "beta_pl": np.unique(ud_beta_pl_map).size,
    }
    patch_indices = get_cutout_from_mask(patch_indices, indices)

    def normalize_array(arr):
        unique_vals, indices = np.unique(arr, return_inverse=True)
        return indices

    patch_indices = jax.tree.map(normalize_array, patch_indices)

    # Define the base parameters and bounds
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
        "beta_pl": -1.0,
    }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    def single_run(noise_id):
        key = jax.random.PRNGKey(noise_id)
        white_noise = f_landscapes.normal(key) * noise_ratio
        white_noise = get_cutout_from_mask(white_noise, indices, axis=1)
        sigma = get_noise_sigma_from_instrument(instrument, nside, stokes_type="QU")
        noise = white_noise * sigma
        noised_d = masked_d + noise

        N = NoiseDiagonalOperator(
            ((sigma * noise_ratio) * noise_ratio) ** 2, _in_structure=masked_d.structure
        )

        guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)
        lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        with ProgressBar(*progress_columns) as p:
            solver = optax.lbfgs()
            final_params, final_state = optimize(
                guess_params,
                negative_log_likelihood_fn,
                solver,
                max_iter=200,
                tol=1e-10,
                lower_bound=lower_bound_tree,
                upper_bound=upper_bound_tree,
                nu=nu,
                N=N,
                d=noised_d,
                progress=p,
                progress_id=noise_id,
                patch_indices=patch_indices,
            )

        cmb_var = spectral_cmb_variance_fn(
            final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices
        )
        nll = negative_log_likelihood_fn(
            final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices
        )
        return {
            "value": cmb_var,
            "NLL": nll,
            "beta_dust": final_params["beta_dust"],
            "temp_dust": final_params["temp_dust"],
            "beta_pl": final_params["beta_pl"],
        }

    results = jax.vmap(single_run)(jnp.arange(nb_noise_sim))

    # Save results and mask
    results["beta_dust_patches"] = patch_indices["beta_dust_patches"]
    results["temp_dust_patches"] = patch_indices["temp_dust_patches"]
    results["beta_pl_patches"] = patch_indices["beta_pl_patches"]
    np.savez(f"{out_folder}/results.npz", **results)
    np.save(f"{out_folder}/cmb_map.npy", cmb_map)
    np.save(f"{out_folder}/mask.npy", mask)
    print("Run complete. Results saved to", out_folder)


if __name__ == "__main__":
    main()
