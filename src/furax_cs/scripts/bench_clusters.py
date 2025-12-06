# Necessary imports
import os

os.environ["EQX_ON_ERROR"] = "nan"
import argparse
import operator
from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns

try:
    from fgbuster import (
        CMB,
        Dust,
        Synchrotron,
        adaptive_comp_sep,
        get_instrument,
    )
except ImportError:
    raise ImportError(
        "FGBuster is required for benchmark comparisons. Install with:\n"
        "  pip install fgbuster\n"
        "or\n"
        "  pip install git+https://github.com/fgbuster/fgbuster.git"
    )

from furax import HomothetyOperator
from furax.obs import negative_log_likelihood, spectral_cmb_variance
from furax.obs.stokes import Stokes
from jax_grid_search import optimize
from jax_healpy.clustering import find_kmeans_clusters
from jax_hpc_profiler import Timer
from jax_hpc_profiler.plotting import plot_weak_scaling

from furax_cs.data.generate_maps import load_from_cache, save_to_cache

jax.config.update("jax_enable_x64", True)


def run_fg_buster(
    nside, cluster_count, freq_maps, dust_nu0, synchrotron_nu0, numpy_timer, max_iter, tol
):
    print(f"Running FGBuster TNC Comp sep nside={nside} cluster_count={cluster_count}...")

    d = Stokes.from_stokes(Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])

    mask = jnp.ones_like(d.q[0]).astype(jnp.int64)

    (indices,) = jnp.where(mask == 1)
    max_centroids = cluster_count

    temp_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(0), max_centroids=max_centroids
    )
    beta_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(1), max_centroids=max_centroids
    )
    beta_pl_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(2), max_centroids=max_centroids
    )
    patch_ids_fg = [
        temp_dust_patch_indices.astype(jnp.int64),
        beta_dust_patch_indices.astype(jnp.int64),
        beta_pl_patch_indices.astype(jnp.int64),
    ]

    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
    freq_maps_fg = jnp.stack([d.q, d.u], axis=1)

    bounds = [(0.0, 5.0), (10.0, 40), (-6.0, 0.0)]
    options = {"disp": False, "gtol": tol, "eps": tol, "maxiter": max_iter, "tol": tol}
    method = "TNC"
    instrument = get_instrument("LiteBIRD")

    freq_maps_fg = np.asarray(freq_maps_fg)
    patch_ids_fg = [np.asarray(p) for p in patch_ids_fg]

    comp_sep = partial(adaptive_comp_sep, bounds=bounds, options=options, method=method, tol=tol)

    result = numpy_timer.chrono_jit(comp_sep, components, instrument, freq_maps_fg, patch_ids_fg)
    for _ in range(2):
        numpy_timer.chrono_fun(comp_sep, components, instrument, freq_maps_fg, patch_ids_fg)

    cmb_q, cmb_u = result.s[0]

    cmb_var = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, (cmb_q, cmb_u)))

    return result.x, cmb_var, result.fun


def run_jax_lbfgs(
    nside, cluster_count, freq_maps, nu, dust_nu0, synchrotron_nu0, jax_timer, max_iter, tol
):
    """Run JAX-based negative log-likelihood."""

    print(f"Running Furax LBGS Comp sep nside={nside} cluster_count={cluster_count}...")

    best_params = {
        "beta_pl": jnp.full((cluster_count,), (-3.0)),
        "beta_dust": jnp.full((cluster_count,), 1.54),
        "temp_dust": jnp.full((cluster_count,), 20.0),
    }

    guess_params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].__hash__()), x.shape),
        best_params,
    )

    d = Stokes.from_stokes(Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)
    mask = jnp.ones_like(d.q[0]).astype(jnp.int64)

    (indices,) = jnp.where(mask == 1)
    max_centroids = cluster_count

    temp_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(0), max_centroids=max_centroids
    )
    beta_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(1), max_centroids=max_centroids
    )
    beta_pl_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(2), max_centroids=max_centroids
    )

    patch_indices = {
        "temp_dust_patches": temp_dust_patch_indices.astype(jnp.int64),
        "beta_dust_patches": beta_dust_patch_indices.astype(jnp.int64),
        "beta_pl_patches": beta_pl_patch_indices.astype(jnp.int64),
    }

    nll = partial(
        negative_log_likelihood,
        nu=nu,
        N=invN,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
    )

    solver = optax.lbfgs()

    def furax_adaptative_comp_sep(guess_params):
        final_params, _ = optimize(
            guess_params,
            nll,
            solver,
            max_iter=max_iter,
            tol=tol,
        )
        return final_params["beta_pl"], final_params

    _, final_params = jax_timer.chrono_jit(furax_adaptative_comp_sep, guess_params)
    for _ in range(2):
        jax_timer.chrono_fun(furax_adaptative_comp_sep, guess_params)

    last_L = nll(final_params)
    cmb_variance = spectral_cmb_variance(
        final_params, nu, invN, d, dust_nu0, synchrotron_nu0, patch_indices
    )

    return final_params, cmb_variance, last_L


def run_jax_tnc(
    nside,
    cluster_count,
    freq_maps,
    nu,
    dust_nu0,
    synchrotron_nu0,
    numpy_timer,
    max_iter,
    tol,
):
    """Run JAX-based negative log-likelihood."""

    print("Running Furax TNC From SciPy Comp sep ")
    print(f"nside={nside} cluster_count={cluster_count} ...")

    best_params = {
        "beta_pl": jnp.full((cluster_count,), (-3.0)),
        "beta_dust": jnp.full((cluster_count,), 1.54),
        "temp_dust": jnp.full((cluster_count,), 20.0),
    }

    guess_params = jax.tree.map_with_path(
        lambda path, x: x + jax.random.normal(jax.random.key(path[0].__hash__()), x.shape),
        best_params,
    )

    d = Stokes.from_stokes(Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    mask = jnp.ones_like(d.q[0]).astype(jnp.int64)

    (indices,) = jnp.where(mask == 1)
    max_centroids = cluster_count

    temp_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(0), max_centroids=max_centroids
    )
    beta_dust_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(1), max_centroids=max_centroids
    )
    beta_pl_patch_indices = find_kmeans_clusters(
        mask, indices, cluster_count, jax.random.PRNGKey(2), max_centroids=max_centroids
    )

    patch_indices = {
        "temp_dust_patches": temp_dust_patch_indices.astype(jnp.int64),
        "beta_dust_patches": beta_dust_patch_indices.astype(jnp.int64),
        "beta_pl_patches": beta_pl_patch_indices.astype(jnp.int64),
    }

    nll = partial(
        negative_log_likelihood,
        nu=nu,
        N=invN,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
    )

    def furax_adaptative_comp_sep(guess_params):
        scipy_solver = jaxopt.ScipyMinimize(
            fun=nll, method="TNC", jit=True, tol=tol, maxiter=max_iter
        )
        result = scipy_solver.run(guess_params)
        return result.params

    final_params = numpy_timer.chrono_jit(furax_adaptative_comp_sep, guess_params)
    for _ in range(2):
        numpy_timer.chrono_fun(furax_adaptative_comp_sep, guess_params)

    last_L = nll(final_params)
    cmb_variance = spectral_cmb_variance(
        final_params, nu, invN, d, dust_nu0, synchrotron_nu0, patch_indices
    )

    return final_params, cmb_variance, last_L


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmarking FGBuster and Furax")
    parser.add_argument(
        "-n",
        "--nsides",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="List of nsides to benchmark",
    )
    parser.add_argument(
        "-cl",
        "--clusters",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="List of cluster counts to benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="comparison",
        help="Output filename prefix for the plots",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10, 6),
        help="Figure size for the plots (width, height)",
    )
    parser.add_argument(
        "-p",
        "--plot-only",
        action="store_true",
        help="Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC",
    )
    parser.add_argument(
        "-c",
        "--cache-run",
        action="store_true",
        help="Run the cache generation step",
    )
    parser.add_argument(
        "-mi",
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations for L-BFGS solver",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-15,
        help="Tolerance for optimization convergence",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instrument = get_instrument("LiteBIRD")
    nu = instrument["frequency"].values
    # stokes_type = 'IQU'
    dust_nu0, synchrotron_nu0 = 150.0, 20.0

    jax_timer = Timer(save_jaxpr=True, jax_fn=True)
    np_timer = Timer(save_jaxpr=False, jax_fn=False)

    if not args.plot_only:
        for nside in args.nsides:
            save_to_cache(nside, sky="c1d1s1", noise=True)

            if args.cache_run:
                continue

            nu, freq_maps = load_from_cache(nside, sky="c1d1s1")

            for cluster_count in args.clusters:
                # Solver mode benchmarking
                print(f"Running solver benchmarking for nside={nside}...")

                # Run Furax TNC from FGBuster
                final_params, cmb_variance, last_L = run_fg_buster(
                    nside,
                    cluster_count,
                    freq_maps,
                    dust_nu0,
                    synchrotron_nu0,
                    np_timer,
                    args.max_iter,
                )
                data = {
                    "final_params": final_params,
                    "cmb_variance": cmb_variance,
                    "last_L": last_L,
                }
                kwargs = {
                    "function": f"FGBuster n={nside}",
                    "precision": "float64",
                    "x": cluster_count,
                    "npz_data": data,
                }
                np_timer.report("runs/CLUSTERS_FGBUSTER.csv", **kwargs)

                # Run JAX LBFGS from Optax
                final_params, cmb_variance, last_L = run_jax_lbfgs(
                    nside,
                    cluster_count,
                    freq_maps,
                    nu,
                    dust_nu0,
                    synchrotron_nu0,
                    jax_timer,
                    args.max_iter,
                )
                data = {
                    "final_params": final_params,
                    "cmb_variance": cmb_variance,
                    "last_L": last_L,
                }
                kwargs = {
                    "function": f"LBFGS n={nside}",
                    "precision": "float64",
                    "x": cluster_count,
                    "npz_data": data,
                }
                jax_timer.report("runs/CLUSTERS_FURAX.csv", **kwargs)

                # Run TNC from SciPy
                final_params, cmb_variance, last_L = run_jax_tnc(
                    nside,
                    cluster_count,
                    freq_maps,
                    nu,
                    dust_nu0,
                    synchrotron_nu0,
                    np_timer,
                    args.max_iter,
                )
                data = {
                    "final_params": final_params,
                    "cmb_variance": cmb_variance,
                    "last_L": last_L,
                }
                kwargs = {
                    "function": f"TNC n={nside}",
                    "precision": "float64",
                    "x": cluster_count,
                    "npz_data": data,
                }
                np_timer.report("runs/CLUSTERS_FURAX.csv", **kwargs)

    # Plot solver results
    if not args.cache_run and args.plot_only:
        plt.rcParams.update({"font.size": 15})
        sns.set_context("paper")

        csv_file = ["runs/CLUSTERS_FGBUSTER.csv", "runs/CLUSTERS_FURAX.csv"]
        solvers = ["TNC n=32", "LBFGS n=32", "FGBuster n=32"]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=solvers,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/CLUSTERS_FGBUSTER_FURAX.png",
        )


if __name__ == "__main__":
    main()
