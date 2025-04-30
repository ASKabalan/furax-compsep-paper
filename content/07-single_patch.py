import os

os.environ["EQX_ON_ERROR"] = "nan"

import argparse
import operator
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from furax._instruments.sky import get_noise_sigma_from_instrument
from furax.comp_sep import negative_log_likelihood, sky_signal
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_grid_search import ProgressBar, optimize
from jax_healpy import get_cutout_from_mask
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Make sure these modules are available in your PYTHONPATH
sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, load_cmb_map, load_fg_map, load_from_cache
from instruments import get_instrument

jax.config.update("jax_enable_x64", True)


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
        default=100,
        help="Number of noise simulations (single run uses 1)",
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

    # Define the output folder and create it if necessary
    out_folder = f"SINGLE_{args.tag}_{args.instrument}_{args.mask}_{int(args.noise_ratio * 100)}"
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
    _, fg_maps = load_fg_map(nside, instrument_name=args.instrument, sky=args.tag)
    cmb_map = load_cmb_map(nside, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    fg_stokes = Stokes.from_stokes(fg_maps[:, 1], fg_maps[:, 2])
    cmb_map_stokes = Stokes.from_stokes(cmb_map[1], cmb_map[2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)
    masked_fg = get_cutout_from_mask(fg_stokes, indices, axis=1)
    masked_cmb = get_cutout_from_mask(cmb_map_stokes, indices)

    # These downgraded maps serve as our patch indices.
    patch_indices = {
        "beta_dust_patches": None,
        "temp_dust_patches": None,
        "beta_pl_patches": None,
    }

    # Define the base parameters and bounds
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

        small_n = (sigma * noise_ratio) ** 2
        small_n = 1.0 if noise_ratio == 0 else small_n

        N = NoiseDiagonalOperator(small_n, _in_structure=masked_d.structure)

        guess_params = jax.tree.map(lambda v: jnp.full((1,), v), base_params)
        # lower_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), lower_bound, max_count)
        # upper_bound_tree = jax.tree.map(lambda v, c: jnp.full((c,), v), upper_bound, max_count)

        with ProgressBar(*progress_columns) as p:
            solver = optax.lbfgs()
            final_params, final_state = optimize(
                guess_params,
                negative_log_likelihood_fn,
                solver,
                max_iter=200,
                tol=1e-10,
                # lower_bound=lower_bound_tree,
                # upper_bound=upper_bound_tree,
                nu=nu,
                N=N,
                d=noised_d,
                progress=p,
                progress_id=noise_id,
                patch_indices=patch_indices,
            )

        s = sky_signal_fn(final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices)
        cmb = s["cmb"]
        cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, cmb))

        cmb_np = jnp.stack([cmb.q, cmb.u])

        nll = negative_log_likelihood_fn(
            final_params, nu=nu, d=noised_d, N=N, patch_indices=patch_indices
        )
        return {
            "value": cmb_var,
            "CMB_O": cmb_np,
            "NLL": nll,
            "beta_dust": final_params["beta_dust"],
            "temp_dust": final_params["temp_dust"],
            "beta_pl": final_params["beta_pl"],
        }

    if not args.best_only:
        results = jax.vmap(single_run)(jnp.arange(nb_noise_sim))
        results["beta_dust_patches"] = np.zeros(1, len(indices)).astype(np.int64)
        results["temp_dust_patches"] = np.zeros(1, len(indices)).astype(np.int64)
        results["beta_pl_patches"] = np.zeros(1, len(indices)).astype(np.int64)
        # Add a new axis to the results so it matches the shape of grid search results
        results = jax.tree.map(lambda x: x[np.newaxis, ...], results)
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
