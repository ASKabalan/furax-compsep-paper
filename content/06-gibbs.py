import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from furax._instruments.sky import (
    get_noise_sigma_from_instrument,
)
from furax.comp_sep import (
    negative_log_likelihood,
    spectral_cmb_variance,
)
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import NoiseDiagonalOperator
from furax.obs.stokes import Stokes
from jax_grid_search import ProgressBar, optimize
from jax_healpy import get_clusters, get_cutout_from_mask
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
from generate_maps import MASK_CHOICES, get_mask, load_from_cache
from instruments import get_instrument
from plotting import plot_grid_search_results

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
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_folder = f"compsep_{args.tag}_{args.instrument}_{args.mask}"
    if args.plot:
        assert os.path.exists(out_folder), "output not found, please run the model first"
        results = np.load(f"{out_folder}/results.npz")
        plot_grid_search_results(
            results, out_folder, best_metric="value", nb_best=12, noise_runs=args.noise_sim
        )
        return

    nside = args.nside
    nb_noise_sim = args.noise_sim
    noise_ratio = args.noise_ratio
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    mask = get_mask(args.mask)
    (indices,) = jnp.where(mask == 1)

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }
    # lower_bound = {
    #    "beta_dust": 1.0,
    #    "temp_dust": 10.0,
    #    "beta_pl": -5.0,
    # }
    # upper_bound = {
    #    "beta_dust": 3.0,
    #    "temp_dust": 30.0,
    #    "beta_pl": -1.0,
    # }

    instrument = get_instrument(args.instrument)
    nu = instrument.frequency
    f_landscapes = FrequencyLandscape(nside, instrument.frequency, "QU")

    spectral_cmb_variance_fn = partial(
        spectral_cmb_variance, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )
    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    _, freqmaps = load_from_cache(nside, noise=False, instrument_name=args.instrument, sky=args.tag)
    d = Stokes.from_stokes(freqmaps[:, 1], freqmaps[:, 2])
    masked_d = get_cutout_from_mask(d, indices, axis=1)

    # Corresponds to PTEP Table page 82

    search_space = {
        "T_d_patches": jnp.array([1, 5, 20, 30, 50, 60, 70, 80]),
        "B_d_patches": jnp.arange(100, 5001, 100),
        "B_s_patches": jnp.array([1, 5, 20, 30, 50, 60, 70, 80]),
    }

    max_count = {
        "beta_dust": np.max(np.array(search_space["B_d_patches"])),
        "temp_dust": np.max(np.array(search_space["T_d_patches"])),
        "beta_pl": np.max(np.array(search_space["B_s_patches"])),
    }
    max_patches = {
        "temp_dust_patches": max_count["temp_dust"],
        "beta_dust_patches": max_count["beta_dust"],
        "beta_pl_patches": max_count["beta_pl"],
    }

    @partial(jax.jit, static_argnums=(5))
    def compute_minimum_variance(
        T_d_patches,
        B_d_patches,
        B_s_patches,
        planck_mask,
        indices,
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
            lambda c, mp: get_clusters(
                mask, indices, c, jax.random.key(0), max_centroids=mp, initial_sample_size=1
            ),
            patch_indices,
            max_patches,
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

        results = jax.vmap(single_run)(jnp.arange(nb_noise_sim))
        results["beta_dust_patches"] = guess_clusters["beta_dust_patches"]
        results["temp_dust_patches"] = guess_clusters["temp_dust_patches"]
        results["beta_pl_patches"] = guess_clusters["beta_pl_patches"]
        return results

    def model(p):
        # Sample discrete hyperparameters
        T_d_index = numpyro.sample(
            "T_d_patches", dist.Categorical(logits=jnp.zeros(len(search_space["T_d_patches"])))
        )
        B_d_index = numpyro.sample(
            "B_d_patches", dist.Categorical(logits=jnp.zeros(len(search_space["B_d_patches"])))
        )
        B_s_index = numpyro.sample(
            "B_s_patches", dist.Categorical(logits=jnp.zeros(len(search_space["B_s_patches"])))
        )

        # Map indices to actual values
        T_d = search_space["T_d_patches"][T_d_index]
        B_d = search_space["B_d_patches"][B_d_index]
        B_s = search_space["B_s_patches"][B_s_index]

        # Compute variance and other values using the function
        results = compute_minimum_variance(T_d, B_d, B_s, mask, indices, p)

        # Log relevant outputs
        numpyro.deterministic("cmb_variance", results["value"])
        numpyro.deterministic("negative_log_likelihood", results["NLL"])
        numpyro.deterministic("beta_dust_patches", results["beta_dust_patches"])
        numpyro.deterministic("temp_dust_patches", results["temp_dust_patches"])
        numpyro.deterministic("beta_pl_patches", results["beta_pl_patches"])
        numpyro.factor("cmb_var_mean", jnp.mean(results["value"]))

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with ProgressBar(*progress_columns) as p:

        def run_inference(num_samples=1, num_warmup=0):
            kernel = DiscreteHMCGibbs(NUTS(model))
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
            mcmc.run(jax.random.PRNGKey(0), p)
            samples = mcmc.get_samples()
            mcmc.print_summary()

            return samples

        # Run inference

        run_inference()


if __name__ == "__main__":
    main()
