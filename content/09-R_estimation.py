import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["EQX_ON_ERROR"] = "nan"

import sys
from functools import partial

import camb
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fgbuster import get_sky
from furax import HomothetyOperator
from furax.comp_sep import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesQU
from jax_grid_search import ProgressBar, optimize
from jax_healpy import combine_masks, from_cutout_to_fullmap
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append("../data")
import scienceplots  # noqa: F401
from instruments import get_instrument

# Set the style for the plots
plt.style.use("science")

out_folder = "plots/"

jax.config.update("jax_enable_x64", True)


def parse_args():
    """
    Parse command-line arguments for benchmark evaluation.

    Returns:
        argparse.Namespace: Parsed arguments including nside, instrument, and run filters.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )

    parser.add_argument("-n", "--nside", type=int, default=64, help="The nside of the map")
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument to use",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
        help="List of run name keywords to filter result folders",
    )
    parser.add_argument(
        "-t",
        "--titles",
        type=str,
        nargs="*",
        help="List of titles for the plots",
    )
    parser.add_argument(
        "-pi",
        "--plot-illustrations",
        action="store_true",
        help="Plot illustrations of the results",
    )
    parser.add_argument(
        "-pv",
        "--plot-validation-curves",
        action="store_true",
        help="Plot validation curves of the results",
    )
    parser.add_argument(
        "-ps",
        "--plot-cl-spectra",
        action="store_true",
        help="Plot spectra of the results one by one",
    )
    parser.add_argument(
        "-pc",
        "--plot-cmb-recon",
        action="store_true",
        help="Plot CMB reconstructions of the results one by one",
    )
    parser.add_argument(
        "-as",
        "--plot-all-spectra",
        action="store_true",
        help="Plot all spectra of the results",
        default=True,
    )
    parser.add_argument(
        "-ac",
        "--plot-all-cmb-recon",
        action="store_true",
        help="Plot all CMB reconstructions of the results",
    )
    parser.add_argument(
        "-a",
        "--plot-all",
        action="store_true",
        help="Plot all results",
    )

    return parser.parse_args()


# ========== Helper Functions ==========
# ======================================


def expand_stokes(stokes_map):
    """
    Promote a StokesI or StokesQU object to a full StokesIQU object.

    Args:
        stokes_map (StokesI | StokesQU | StokesIQU): Input Stokes map.

    Returns:
        StokesIQU: Complete IQU map.
    """
    if isinstance(stokes_map, StokesIQU):
        return stokes_map

    zeros = np.zeros(shape=stokes_map.shape, dtype=stokes_map.dtype)

    if isinstance(stokes_map, StokesI):
        return StokesIQU(stokes_map, zeros, zeros)
    elif isinstance(stokes_map, StokesQU):
        return StokesIQU(zeros, stokes_map.q, stokes_map.u)


def filter_constant_param(input_dict, indx):
    """
    Filter a dictionary of arrays using an index.

    Args:
        input_dict (dict): Input data dictionary.
        indx (int): Index to extract.

    Returns:
        dict: Dictionary with extracted entries.
    """
    return jax.tree.map(lambda x: x[indx], input_dict)


def sort_results(results, key):
    """
    Sort a result dictionary by a specific key.

    Args:
        results (dict): Dictionary of results.
        key (str): Key to sort by.

    Returns:
        dict: Sorted result dictionary.
    """
    indices = np.argsort(results[key])
    return jax.tree.map(lambda x: x[indices], results)


# ========== R Estimation Functions ==========
# ============================================


def log_likelihood(r, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    """
    Compute the Gaussian log-likelihood for the tensor-to-scalar ratio r.

    Args:
        r (float): Trial r value.
        ell_range (np.ndarray): Array of multipole moments.
        cl_obs (np.ndarray): Observed BB spectrum.
        cl_bb_r1 (np.ndarray): Template BB spectrum for r=1.
        cl_bb_lens (np.ndarray): Lensing contribution.
        cl_noise (np.ndarray): Statistical residual power.
        f_sky (float): Sky fraction observed.

    Returns:
        float: Log-likelihood value.
    """
    cl_model = r.reshape(-1, 1) * cl_bb_r1 + cl_bb_lens + cl_noise
    term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term, axis=1)


def estimate_r(cl_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    """
    Estimate best-fit r and its uncertainties using likelihood maximization.

    Args:
        cl_obs (np.ndarray): Observed Cl_BB.
        ell_range (np.ndarray): Multipole range.
        cl_bb_r1 (np.ndarray): BB spectrum template for r=1.
        cl_bb_lens (np.ndarray): Lensing spectrum.
        cl_noise (np.ndarray): Noise (statistical residual).
        f_sky (float): Observed sky fraction.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray]:
            r_best, lower_sigma, upper_sigma, r_grid, likelihood_vals
    """
    r_grid = np.linspace(-0.005, 0.005, 1000)
    logL = log_likelihood(r_grid, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky)
    finite_logL = logL[np.isfinite(logL)]
    finite_r = r_grid[np.isfinite(logL)]
    L = np.exp(finite_logL - np.max(finite_logL))
    r_best = finite_r[np.argmax(L)]

    rs_pos, L_pos = finite_r[finite_r > r_best], L[finite_r > r_best]
    rs_neg, L_neg = finite_r[finite_r < r_best], L[finite_r < r_best]
    cum_pos = np.cumsum(L_pos) / np.sum(L_pos)
    cum_neg = np.cumsum(L_neg[::-1]) / np.sum(L_neg)

    sigma_pos = rs_pos[np.argmin(np.abs(cum_pos - 0.68))] - r_best if len(rs_pos) > 0 else 0
    sigma_neg = r_best - rs_neg[::-1][np.argmin(np.abs(cum_neg - 0.68))] if len(rs_neg) > 0 else 0
    return r_best, sigma_neg, sigma_pos, finite_r, L


def get_camb_templates(nside):
    """
    Generate Cl_BB theoretical spectra using CAMB for r=1 and lensing.

    Args:
        nside (int): HEALPix resolution parameter.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ell_range, cl_bb_r1, cl_bb_lens
    """
    pars = camb.set_params(
        ombh2=0.022,
        omch2=0.12,
        tau=0.054,
        As=2e-9,
        ns=0.965,
        cosmomc_theta=1.04e-2,
        r=1,  # Pourquoi r = 1 ?
        DoLensing=True,
        WantTensors=True,
        Want_CMB_lensing=True,
        lmax=1024,
    )
    pars_0 = camb.set_params(
        ombh2=0.022,
        omch2=0.12,
        tau=0.054,
        As=2e-9,
        ns=0.965,
        cosmomc_theta=1.04e-2,
        r=0,  # Pourquoi r = 1 ?
        DoLensing=True,
        WantTensors=True,
        Want_CMB_lensing=True,
        lmax=1024,
    )
    # ERROR R0
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=1024)
    cl_bb_r1_full, cl_bb_total = powers["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_full = cl_bb_total - cl_bb_r1_full

    ell_min, ell_max = 2, nside * 2 + 2
    ell_range = np.arange(ell_min, ell_max)
    coeff = ell_range * (ell_range + 1) / (2 * np.pi)
    cl_bb_r1 = cl_bb_r1_full[ell_range] / coeff
    cl_bb_lens = cl_bb_lens_full[ell_range] / coeff

    # get r0
    results_0 = camb.get_results(pars_0)
    powers_0 = results_0.get_cmb_power_spectra(pars_0, CMB_unit="muK", lmax=1024)
    cl_bb_r0_full, cl_bb_total = powers_0["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_r0_full = cl_bb_total - cl_bb_r0_full

    cl_bb_r0 = cl_bb_r0_full[ell_range] / coeff
    cl_bb_r0_lens = cl_bb_lens_r0_full[ell_range] / coeff

    return ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, cl_bb_r0_lens


# ========== CL COMPUTATION FUNCTIONS ==========
# ============================================


def compute_cl_bb_sum_partial(cmb_out, patches, nside, ell_range, fsky, results, result_file):
    if results.get("CL_BB_SUM") is not None and True:
        print("Using CL_BB_SUM from results")
        CL_BB_SUM = results["CL_BB_SUM"]
        return CL_BB_SUM

    cmb_out_full_sky = from_cutout_to_fullmap(cmb_out, patches, nside, axis=1)
    cmb_out_full_sky = expand_stokes(cmb_out_full_sky)
    cmb_out_full_sky = np.stack(
        [cmb_out_full_sky.i, cmb_out_full_sky.q, cmb_out_full_sky.u], axis=1
    )

    cl_list = []
    for i in range(cmb_out_full_sky.shape[0]):
        cl = hp.anafast(cmb_out_full_sky[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    CL_BB_SUM = np.sum(cl_list, axis=1) / fsky  # shape (noise realisations,)
    results_from_file = dict(np.load(result_file))
    results_from_file["CL_BB_SUM"] = CL_BB_SUM
    np.savez(result_file, **results_from_file)
    return CL_BB_SUM


def compute_w(nu, d, results, result_file):
    """
    Apply the linear component separation operator W to the input sky.

    Args:
        nu (np.ndarray): Instrument frequencies.
        d (Stokes): Input sky without CMB.
        results (dict): Fitted spectral parameters.
        result_file (str): Name of the result file.

    Returns:
        StokesQU: Reconstructed CMB map.
    """
    if results.get("W_D_FG") is not None and True:
        print("Using W_D_FG from results")
        W = results["W_D_FG"]
        W = Stokes.from_stokes(Q=W[0], U=W[1])
        return W

    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    patches = {k: results[k] for k in ["beta_dust_patches", "beta_pl_patches", "temp_dust_patches"]}
    max_count = {
        "beta_dust": patches["beta_dust_patches"].size,
        "temp_dust": patches["temp_dust_patches"].size,
        "beta_pl": patches["beta_pl_patches"].size,
    }

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }

    guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)

    N = HomothetyOperator(1.0, _in_structure=d.structure)

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

    with ProgressBar(*progress_columns) as p:
        solver = optax.lbfgs()
        final_params, final_state = optimize(
            guess_params,
            negative_log_likelihood_fn,
            solver,
            max_iter=1000,
            tol=1e-10,
            progress=p,
            progress_id=0,
            # lower_bound=lower_bound_tree,
            # upper_bound=upper_bound_tree,
            nu=nu,
            N=N,
            d=d,
            patch_indices=patches,
        )

    def W(p):
        N = HomothetyOperator(1.0, _in_structure=d.structure)
        return sky_signal(
            p, nu, N, d, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patches
        )["cmb"]

    W = W(final_params)

    results_from_file = dict(np.load(result_file))
    W_numpy = np.stack([W.q, W.u], axis=0)
    results_from_file["W_D_FG"] = W_numpy[np.newaxis, ...]
    np.savez(result_file, **results_from_file)
    return W


def compute_systematic_res(Wd_cmb, fsky, ell_range):
    """
    Compute the BB spectrum of the systematic residual map.

    Args:
        Wd_cmb (StokesQU): CMB estimated from foreground-only data.
        ell_range (np.ndarray): Multipole moments.

    Returns:
        np.ndarray: BB power spectrum of systematics.
    """
    Wd_cmb = expand_stokes(Wd_cmb)
    Wd_cmb = np.stack([Wd_cmb.i, Wd_cmb.q, Wd_cmb.u], axis=0)  # shape (3 , masked_npix)
    Wn_cl = hp.sphtfunc.anafast(Wd_cmb)
    Wn_cl = Wn_cl[2][ell_range]  # shape (len(ell_range),)
    return Wn_cl / fsky  # shape (len(ell_range),)


def compute_total_res(s_hat, s_true, fsky, ell_range):
    """
    Compute average BB residual spectrum from multiple noisy realizations.

    Args:
        s_hat (StokesQU): Reconstructed CMB (n_sims, ...)
        s_true (StokesQU): Ground truth CMB map.
        ell_range (np.ndarray): Multipole moments.

    Returns:
        np.ndarray: Residual BB spectrum.
    """
    s_hat = expand_stokes(s_hat)
    s_hat = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)

    res = np.where(s_hat == hp.UNSEEN, hp.UNSEEN, s_hat - s_true[np.newaxis, ...])
    cl_list = []
    for i in range(res.shape[0]):
        cl = hp.anafast(res[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    return np.mean(cl_list, axis=0) / fsky  # shape (len(ell_range),)


def compute_cl_bb_sum(cmb_out, fsky, ell_range):
    cmb_out = expand_stokes(cmb_out)
    cmb_out = np.stack([cmb_out.i, cmb_out.q, cmb_out.u], axis=1)

    cl_list = []
    for i in range(cmb_out.shape[0]):
        cl = hp.anafast(cmb_out[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    CL_BB_SUM = np.sum(cl_list, axis=1) / fsky  # shape (noise realisations,)
    return CL_BB_SUM


def compute_cl_obs_bb(cl_total_res, cl_bb_lens):
    return cl_total_res + cl_bb_lens


# def compute_cl_obs_bb(s_hat, ell_range):
#    s_hat = expand_stokes(s_hat)
#    s_hat = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)
#
#    cl_list = []
#    for i in range(s_hat.shape[0]):
#        cl = hp.anafast(s_hat[i])
#        cl_list.append(cl[2][ell_range])
#
#    return np.mean(cl_list, axis=0)


def compute_cl_true_bb(s, ell_range):
    """
    Compute the average observed Cl_BB from reconstructed maps.

    Args:
        s_hat (StokesQU): Reconstructed maps (n_sims, ...).
        ell_range (np.ndarray): Multipole moments.

    Returns:
        np.ndarray: Averaged BB spectrum.
    """

    # coeff = ell_range * (ell_range + 1) / (2 * np.pi)

    cl = hp.anafast(s)  # shape (6, lmax+1)

    return cl[2][ell_range]  # shape (len(ell_range),)


# ========== Illustrations ==========
# ============================================


def params_to_maps(run_data, previous_mask_size):
    B_d_patches = run_data["beta_dust_patches"]
    T_d_patches = run_data["temp_dust_patches"]
    B_s_patches = run_data["beta_pl_patches"]

    B_d = run_data["beta_dust"]
    T_d = run_data["temp_dust"]
    B_s = run_data["beta_pl"]

    B_d = B_d.mean(axis=0)[B_d_patches]
    T_d = T_d.mean(axis=0)[T_d_patches]
    B_s = B_s.mean(axis=0)[B_s_patches]

    params = {"beta_dust": B_d, "temp_dust": T_d, "beta_pl": B_s}
    patches = {
        "beta_dust_patches": B_d_patches,
        "temp_dust_patches": T_d_patches,
        "beta_pl_patches": B_s_patches,
    }

    def normalize_array(arr):
        unique_vals, indices = np.unique(arr, return_inverse=True)
        return indices

    patches = jax.tree.map(normalize_array, patches)
    patches = jax.tree.map(lambda x, p: x + p, patches, previous_mask_size)
    previous_mask_size = jax.tree.map(
        lambda x, p: p + np.unique(x).size, patches, previous_mask_size
    )

    return params, patches, previous_mask_size


def plot_params_patches(name, params, patches):
    # Params on a figure
    _ = plt.figure(figsize=(7.5, 13))

    keys = ["beta_dust", "temp_dust", "beta_pl"]
    names = ["$\\beta_d$", "$T_d$", "$\\beta_s$"]

    for i, (key, param_name) in enumerate(zip(keys, names)):
        param_map = params[key]
        hp.mollview(
            param_map,
            title=f"{name} {param_name}",
            sub=(3, 1, i + 1),
            bgcolor=(0.0,) * 4,
            cbar=True,
        )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/params_{name}.pdf", transparent=True, dpi=1200)

    # Patches on a figure
    _ = plt.figure(figsize=(7.5, 13))

    np.random.seed(0)

    # Shuffle labels in each patch
    def shuffle_labels(arr):
        unique_vals = np.unique(arr[arr != hp.UNSEEN])  # Ignore UNSEEN
        shuffled_vals = np.random.permutation(unique_vals)

        # Create mapping dict
        mapping = dict(zip(unique_vals, shuffled_vals))

        # Vectorized mapping
        shuffled_arr = np.vectorize(lambda x: mapping.get(x, hp.UNSEEN))(arr)
        return shuffled_arr.astype(np.float64)

    patches = jax.tree.map(shuffle_labels, patches)

    keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    names = ["$\\beta_d$ Patches", "$T_d$ Patches", "$\\beta_s$ Patches"]

    for i, (key, patch_name) in enumerate(zip(keys, names)):
        patch_map = patches[key]
        hp.mollview(
            patch_map,
            title=f"{name} {patch_name}",
            sub=(3, 1, i + 1),
            bgcolor=(0.0,) * 4,
            cbar=True,
        )
    plt.tight_layout()
    plt.savefig(f"{out_folder}/patches_{name}.pdf", transparent=True, dpi=1200)


def plot_validation_curves(name, updates_history, value_history):
    updates_history = np.array(updates_history)
    value_history = np.array(value_history)

    n_runs = updates_history.shape[0]
    ncols = 2  # One column for updates, one for values
    nrows = int(np.ceil(n_runs))  # One row per run

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))

    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)  # Ensure 2D indexing

    for i in range(n_runs):
        updates = updates_history[i].mean(axis=0)
        values = value_history[i].mean(axis=0)

        # Remove invalid zeros (or could use np.isfinite for NaNs)
        valid_mask = values != 0.0
        updates = updates[: len(valid_mask)]
        values = values[valid_mask]
        updates = updates[valid_mask]
        indx = np.arange(len(values))

        axs[i, 0].plot(indx, updates, label=f"Run {i + 1} Updates")
        axs[i, 0].set_title(f"{name} - Updates History Run {i + 1}")
        axs[i, 0].set_xlabel("Iteration")
        axs[i, 0].set_ylabel("Update Norm")
        axs[i, 0].grid(True)
        axs[i, 0].legend()

        axs[i, 1].plot(indx, values, label=f"Run {i + 1} NLL")
        axs[i, 1].set_title(f"{name} - NLL History Run {i + 1}")
        axs[i, 1].set_xlabel("Iteration")
        axs[i, 1].set_ylabel("Negative Log-Likelihood")
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_folder}/validation_curves_{name}.pdf", transparent=True, dpi=1200)


# ========== Plot All runs ===================
# ============================================


def get_min_variance(cmb_map):
    seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
    cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
    variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
    variance = sum(jax.tree.leaves(variance))
    argmin = jnp.argmin(variance)
    return jax.tree.map(lambda x: x[argmin], cmb_map)


def plot_all_cmb(names, cmb_pytree_list):
    nb_cmb = len(cmb_pytree_list)

    diff_all = []

    for cmb_pytree in cmb_pytree_list:
        cmb_recon = get_min_variance(cmb_pytree["cmb_recon"])

        diff_q = cmb_pytree["cmb"].q - cmb_recon.q
        diff_u = cmb_pytree["cmb"].u - cmb_recon.u

        unseen_mask_q = cmb_pytree["cmb"].q == hp.UNSEEN
        unseen_mask_u = cmb_pytree["cmb"].u == hp.UNSEEN

        diff_q = np.where(unseen_mask_q, np.nan, diff_q)
        diff_u = np.where(unseen_mask_u, np.nan, diff_u)

        diff_all.append((diff_q, diff_u))

    # Global min/max for shared color scale
    all_values = [val for pair in diff_all for val in pair]
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    plt.figure(figsize=(10, 3.5 * nb_cmb))

    for i, (name, (diff_q, diff_u)) in enumerate(zip(names, diff_all)):
        # Q map
        hp.mollview(
            diff_q,
            title=rf"Difference (Q) - {name} $\mu_k$",
            sub=(nb_cmb, 2, 2 * i + 1),
            min=vmin,
            max=vmax,
            cmap="viridis",
            bgcolor=(0.0,) * 4,
            cbar=True,
            notext=True,
        )
        # U map
        hp.mollview(
            diff_u,
            title=rf"Difference (U) - {name} $\mu_k$",
            sub=(nb_cmb, 2, 2 * i + 2),
            min=vmin,
            max=vmax,
            cmap="viridis",
            bgcolor=(0.0,) * 4,
            cbar=True,
            notext=True,
        )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/cmb_recon.pdf", transparent=True, dpi=1200)


def plot_all_variances(names, cmb_pytree_list):
    def get_all_variances(cmb_map):
        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance = sum(jax.tree.leaves(variance))  # shape (100,)
        return variance

    fig, axs = plt.subplots(3, 1, figsize=(7, 15), sharex=False)

    metrics = {
        "Variance of Reconstructed CMB (Q + U)": [],
        "Negative Log-Likelihood": [],
        r"$\sum C_\ell^{BB}$": [],
    }
    B_s_counts = {}

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        metrics["Variance of Reconstructed CMB (Q + U)"].append(
            (name, get_all_variances(cmb_pytree["cmb_recon"]))
        )
        metrics["Negative Log-Likelihood"].append((name, np.array(cmb_pytree["nll_summed"])))
        metrics[r"$\sum C_\ell^{BB}$"].append((name, np.array(cmb_pytree["cl_bb_sum"])))
        # Count of synchrotron patches
        patches = cmb_pytree["patches_map"]
        B_s_patches = patches["temp_dust_patches"]
        B_s_count = np.unique(B_s_patches[B_s_patches != hp.UNSEEN]).size
        B_s_counts[name] = B_s_count

    for ax, (title, entries) in zip(axs, metrics.items()):
        for i, (name, values) in enumerate(entries):
            B_s_count = B_s_counts.get(name, 0)
            # Assign a unique color
            color = plt.cm.tab10(i % 10)
            label = f"{name} B_s count {B_s_count}"
            ax.hist(
                values,
                bins=20,
                alpha=0.5,
                label=label,
                color=color,
                edgecolor="black",
                histtype="stepfilled",
            )
            mean_val = np.mean(values)
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=2, label=f"Mean of {name}")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize="small")

    axs[-1].set_xlabel("Metric Value")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/metric_distributions_histogram.pdf", transparent=True, dpi=1200)


def plot_all_cl_residuals(names, cl_pytree_list):
    _ = plt.figure(figsize=(8, 6))

    if len(cl_pytree_list) == 0:
        print("No results")
        return

    cl_bb_r1 = cl_pytree_list[0]["cl_bb_r1"]
    # cl_bb_true = cl_pytree_list[0]["cl_true"]
    ell_range = cl_pytree_list[0]["ell_range"]
    cl_bb_lens = cl_pytree_list[0]["cl_bb_lens"]
    coeff = ell_range * (ell_range + 1) / (2 * np.pi)

    plt.plot(
        ell_range,
        cl_bb_r1 * coeff * 1e-3,
        label=r"$C_\ell^{\mathrm{BB}}(r=10^{-3})$",
        color="black",
        linewidth=2,
    )
    # plt.plot(
    #    ell_range,
    #    cl_bb_true * coeff,
    #    label=r"$C_\ell^{\mathrm{true}}$",
    #    color="purple",
    #    linestyle="--",
    # )
    plt.plot(
        ell_range,
        cl_bb_lens * coeff,
        label=r"$C_\ell^{\mathrm{lens}}$",
        color="black",
        linestyle=":",
        linewidth=2,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, cl_pytree) in enumerate(zip(names, cl_pytree_list)):
        color = colors[i % len(colors)]
        # plt.plot(
        #    ell_range,
        #    cl_pytree["cl_bb_obs"] * coeff,
        #    label=rf"{name} $C_\ell^{{\mathrm{{obs}}}}$",
        #    color=color,
        #    linestyle="-",
        # )
        # plt.plot(
        #    ell_range,
        #    cl_pytree["cl_total_res"] * coeff,
        #    label=rf"{name} $C_\ell^{{\mathrm{{res}}}}$",
        #    color=color,
        #    linestyle="--",
        # )
        plt.plot(
            ell_range,
            cl_pytree["cl_syst_res"] * coeff,
            label=rf"{name} $C_\ell^{{\mathrm{{syst}}}}$",
            color=color,
            linestyle="-",
        )
        plt.plot(
            ell_range,
            cl_pytree["cl_stat_res"] * coeff,
            label=rf"{name} $C_\ell^{{\mathrm{{stat}}}}$",
            color=color,
            linestyle=":",
        )

    plt.title("BB Power Spectra (All Runs)")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [1e-2 $\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out_folder}/bb_spectra.pdf", transparent=True, dpi=1200)


def plot_all_r_estimation(names, r_pytree_list):
    plt.figure(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, r_data) in enumerate(zip(names, r_pytree_list)):
        r_grid = r_data["r_grid"]
        L_vals = r_data["L_vals"]
        r_best = r_data["r_best"]
        sigma_r_neg = r_data["sigma_r_neg"]
        sigma_r_pos = r_data["sigma_r_pos"]

        color = colors[i % len(colors)]
        likelihood = L_vals / L_vals.max()

        # Plot likelihood curve
        plt.plot(
            r_grid,
            likelihood,
            label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",  # noqa : E501
            color=color,
        )

        # Fill the ±1σ region
        plt.fill_between(
            r_grid,
            0,
            likelihood,
            where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
            color=color,
            alpha=0.2,
        )

        # Add vertical dotted line at best-fit r
        plt.axvline(
            x=r_best,
            color=color,
            linestyle="--",
            alpha=0.7,
        )

    plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.7, label="True r=0")

    plt.title("Likelihood Curves for $r$ (All Runs)")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize="medium")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/bb_spectra_and_r_likelihood.pdf", transparent=True, dpi=1200)


# ========== Plot Single Run ===================
# ==============================================


def plot_cmb_reconsturctions(name, cmb_stokes, cmb_recon):
    def mse(a, b):
        seen_x = jax.tree.map(lambda x: x[x != hp.UNSEEN], a)
        seen_y = jax.tree.map(lambda x: x[x != hp.UNSEEN], b)
        return jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), seen_x, seen_y)

    cmb_recon_min = get_min_variance(cmb_recon)
    mse_cmb = mse(cmb_recon_min, cmb_stokes)
    cmb_recon_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_recon_min)
    cmb_input_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_stokes)
    print("======================")
    print(f"MSE CMB: {mse_cmb}")
    print(f"Reconstructed CMB variance: {cmb_recon_var}")
    print(f"Input CMB variance: {cmb_input_var}")
    print("======================")
    unseen_mask = cmb_recon_min.q == hp.UNSEEN
    diff_q = cmb_recon_min.q - cmb_stokes.q
    diff_q = np.where(unseen_mask, hp.UNSEEN, diff_q)

    unseen_mask = cmb_recon_min.u == hp.UNSEEN
    diff_u = cmb_recon_min.u - cmb_stokes.u
    diff_u = np.where(unseen_mask, hp.UNSEEN, diff_u)

    _ = plt.figure(figsize=(12, 8))
    hp.mollview(cmb_recon_min.q, title="Reconstructed CMB (Q)", sub=(2, 3, 1), bgcolor=(0.0,) * 4)
    hp.mollview(cmb_stokes.q, title="Input CMB Map (Q)", sub=(2, 3, 2), bgcolor=(0.0,) * 4)
    hp.mollview(
        diff_q,
        title="Difference (Q)",
        sub=(2, 3, 3),
        bgcolor=(0.0,) * 4,
    )
    hp.mollview(cmb_recon_min.u, title="Reconstructed CMB (U)", sub=(2, 3, 4), bgcolor=(0.0,) * 4)
    hp.mollview(cmb_stokes.u, title="Input CMB Map (U)", sub=(2, 3, 5), bgcolor=(0.0,) * 4)
    hp.mollview(
        diff_u,
        title="Difference (U)",
        sub=(2, 3, 6),
        bgcolor=(0.0,) * 4,
    )
    plt.title(f"{name} CMB Reconstruction")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/cmb_recon_{name}.pdf", transparent=True, dpi=1200)


def plot_cl_residuals(
    name,
    cl_bb_obs,
    cl_syst_res,
    cl_total_res,
    cl_stat_res,
    cl_bb_r1,
    cl_bb_r0,
    cl_bb_lens,
    cl_true,
    ell_range,
):
    _ = plt.figure(figsize=(12, 8))

    coeff = ell_range * (ell_range + 1) / (2 * np.pi)

    # --- Power Spectrum Plot ---
    plt.plot(ell_range, cl_bb_obs * coeff, label=r"$C_\ell^{\mathrm{obs}}$", color="green")
    plt.plot(ell_range, cl_total_res * coeff, label=r"$C_\ell^{\mathrm{res}}$", color="black")
    plt.plot(ell_range, cl_syst_res * coeff, label=r"$C_\ell^{\mathrm{syst}}$", color="blue")
    plt.plot(ell_range, cl_stat_res * coeff, label=r"$C_\ell^{\mathrm{stat}}$", color="orange")
    plt.plot(ell_range, cl_bb_r1 * coeff, label=r"$C_\ell^{\mathrm{BB}}(r=1)$", color="red")
    plt.plot(ell_range, cl_bb_r0 * coeff, label=r"$C_\ell^{\mathrm{BB}}(r=0)$", color="orange")
    plt.plot(
        ell_range,
        cl_true * coeff,
        label=r"$C_\ell^{\mathrm{true}}$",
        color="purple",
        linestyle="--",
    )
    plt.plot(
        ell_range,
        cl_bb_lens * coeff,
        label=r"$C_\ell^{\mathrm{lens}}$",
        color="purple",
        linestyle=":",
    )

    plt.title(f"{name} BB Power Spectra")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [1e-2 $\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_folder}/bb_spectra_{name}.pdf", transparent=True, dpi=1200)


def plot_r_estimator(
    name,
    r_best,
    sigma_r_neg,
    sigma_r_pos,
    r_grid,
    L_vals,
):
    plt.figure(figsize=(12, 8))

    # Normalize likelihoods
    likelihood = L_vals / np.max(L_vals)

    # Plot reconstructed likelihood
    plt.plot(
        r_grid,
        likelihood,
        label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",
        color="purple",
        linewidth=2,
    )
    plt.fill_between(
        r_grid,
        0,
        likelihood,
        where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
        color="purple",
        alpha=0.2,
    )
    plt.axvline(r_best, color="purple", linestyle="--", alpha=0.8)

    # Plot true likelihood
    plt.axvline(0.0, color="purple", linestyle="--", alpha=0.8, label="True r=0")
    # Labels and plot config
    plt.title(f"{name} Likelihood vs $r$")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save + Show
    plt.savefig(f"{out_folder}/bb_spectra_and_r_likelihood_{name}.pdf", transparent=True, dpi=1200)

    # Print
    print(f"Estimated r (Reconstructed): {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


def plot_results(name, filtered_results, nside, instrument, args):
    """
    Load, combine, and analyze PTEP results, plotting spectra and r-likelihood.

    Args:
        filtered_results (list[str]): List of result directories.
        nside (int): HEALPix resolution.
        instrument (Instrument): Instrument object.
    """
    if len(filtered_results) == 0:
        print("No results")
        return

    cmb_recons, cmb_maps, masks, NLLs = [], [], [], []
    indices_list, w_d_list, cl_bb_sum_list = [], [], []
    params_list, patches_list = [], []
    updates_history, value_history = [], []
    ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, cl_bb_lens_r0 = get_camb_templates(nside=64)

    previous_mask_size = {
        "beta_dust_patches": 0,
        "temp_dust_patches": 0,
        "beta_pl_patches": 0,
    }

    for folder in filtered_results:
        run_data = dict(np.load(f"{folder}/results.npz"))
        best_params = dict(np.load(f"{folder}/best_params.npz"))
        mask = np.load(f"{folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)
        f_sky = mask.sum() / len(mask)

        # Get best run data
        run_data = jax.tree.map(lambda x: x[0], run_data)
        NLL = run_data["NLL"]

        cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
        fg_map = Stokes.from_stokes(
            Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
        )
        cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])
        wd = compute_w(instrument.frequency, fg_map, run_data, result_file=f"{folder}/results.npz")
        cl_bb_sum = compute_cl_bb_sum_partial(
            cmb_recon, indices, 64, ell_range, f_sky, run_data, result_file=f"{folder}/results.npz"
        )

        if args.plot_illustrations:
            params, patches, previous_mask_size = params_to_maps(run_data, previous_mask_size)
            params_list.append(params)
            patches_list.append(patches)

        if args.plot_validation_curves:
            updates_history.append(run_data["update_history"][..., 0])
            value_history.append(run_data["update_history"][..., 1])

        cmb_recons.append(cmb_recon)
        cl_bb_sum_list.append(cl_bb_sum)
        cmb_maps.append(cmb_true)
        w_d_list.append(wd)
        masks.append(mask)
        indices_list.append(indices)
        NLLs.append(NLL)

    full_mask = np.logical_or.reduce(masks)

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)
    wd = combine_masks(w_d_list, indices_list, nside)

    NLL_summed = np.sum(NLLs, axis=0)

    if args.plot_illustrations:
        params_map = combine_masks(params_list, indices_list, nside)
        patches_map = combine_masks(patches_list, indices_list, nside)
        plot_params_patches(name, params_map, patches_map)
    else:
        patches_map = {
            "beta_dust_patches": np.zeros(hp.nside2npix(nside)),
            "temp_dust_patches": np.zeros(hp.nside2npix(nside)),
            "beta_pl_patches": np.zeros(hp.nside2npix(nside)),
        }

    if args.plot_validation_curves:
        plot_validation_curves(
            name,
            updates_history,
            value_history,
        )

    s_true = get_sky(64, "c1d1s1").components[0].map.value

    if args.plot_cmb_recon:
        plot_cmb_reconsturctions(name, cmb_stokes, combined_cmb_recon)

    f_sky = full_mask.sum() / len(full_mask)
    # Compute the systematic residuals Cl_syst = Cl(W(d_no_cmb))
    cl_syst_res = compute_systematic_res(wd, f_sky, ell_range)
    print(f"maximum cl_syst_res: {np.max(cl_syst_res)}")
    # Compute the total residuals Cl_res = <CL(s_hat - s_true)>n
    cl_total_res = compute_total_res(combined_cmb_recon, s_true, f_sky, ell_range)
    # Compute the statistical residuals Cl_stat = CL_res - CL_syst
    cl_stat_res = jnp.abs(cl_total_res - cl_syst_res)
    # Compute cl_true
    cl_true = compute_cl_true_bb(s_true, ell_range)
    # Compute observed Cl_obs = <CL(s_hat)>
    cl_bb_obs = compute_cl_obs_bb(cl_total_res, cl_bb_lens)
    # cl_bb_obs = compute_cl_obs_bb(combined_cmb_recon , ell_range)
    # cl_bb_obs = cl_bb_lens + cl_total_res
    cl_bb_sum = compute_cl_bb_sum(combined_cmb_recon, f_sky, ell_range)

    if args.plot_cl_spectra:
        plot_cl_residuals(
            name,
            cl_bb_obs,
            cl_syst_res,
            cl_total_res,
            cl_stat_res,
            cl_bb_r1,
            cl_bb_r0,
            cl_bb_lens,
            cl_true,
            ell_range,
        )

    # --- Likelihood Plot ---
    f_sky = full_mask.sum() / len(full_mask)
    # compute r from obs
    r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
        cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_stat_res, f_sky
    )

    if args.plot_cl_spectra:
        plot_r_estimator(
            name,
            r_best,
            sigma_r_neg,
            sigma_r_pos,
            r_grid,
            L_vals,
        )

    cmb_pytree = {
        "cmb": cmb_stokes,
        "cmb_recon": combined_cmb_recon,
        "patches_map": patches_map,
        "cl_bb_sum": cl_bb_sum,
        "nll_summed": NLL_summed,
    }
    cl_pytree = {
        "cl_bb_r1": cl_bb_r1,
        "cl_true": cl_true,
        "ell_range": ell_range,
        "cl_bb_obs": cl_bb_obs,
        "cl_bb_lens": cl_bb_lens,
        "cl_syst_res": cl_syst_res,
        "cl_total_res": cl_total_res,
        "cl_stat_res": cl_stat_res,
    }
    r_pytree = {
        "r_best": r_best,
        "sigma_r_neg": sigma_r_neg,
        "sigma_r_pos": sigma_r_pos,
        "r_grid": r_grid,
        "L_vals": L_vals,
    }

    return cmb_pytree, cl_pytree, r_pytree


# ========== Main Function ================
# =========================================


def parse_filter_kw(kw_string):
    """
    Parse a string like 'A_(B|C)_D' into a list of sets (ORs within, AND across).
    """
    groups = kw_string.split("_")
    parsed = []
    for group in groups:
        if group.startswith("(") and group.endswith(")"):
            options = group[1:-1].split("|")
            parsed.append(set(options))
        else:
            parsed.append({group})
    return parsed


def matches_filter(name_parts, filter_groups):
    """
    Check if name_parts satisfies the AND-of-OR filter groups.
    """
    return all(any(option in name_parts for option in group) for group in filter_groups)


def main():
    """
    Entry point for evaluating and plotting r estimation from multiple runs.
    """
    args = parse_args()
    nside = args.nside
    instrument = get_instrument(args.instrument)
    result_folder = "../results/"
    print("Loading data...")
    results = os.listdir(result_folder)
    results_kw = {name: name.split("_") for name in results}

    os.makedirs(out_folder, exist_ok=True)

    if args.plot_all:
        args.plot_cmb_recon = True
        args.plot_cl_spectra = True
        args.plot_all_cmb_recon = True
        args.plot_all_spectra = True
        args.plot_all_r_estimation = True
        args.plot_validation_curves = True
        args.plot_illustrations = True

    results_to_plot = []
    for filter_expr in args.runs:  # args.runs contains strings like "A_(B|C)_D"
        filter_groups = parse_filter_kw(filter_expr)
        group = []
        for result_name, res_kw in results_kw.items():
            if matches_filter(res_kw, filter_groups):
                group.append(os.path.join(result_folder, result_name))
        if group:
            results_to_plot.append(group)

    print("Results to plot: ", results_to_plot)
    assert len(args.titles) == len(results_to_plot), "Number of names must match number of results"

    cmb_pytree_list = []
    cl_pytree_list = []
    r_pytree_list = []
    for name, group_results in zip(args.titles, results_to_plot):
        cmb_pytree, cl_pytree, r_pytree = plot_results(name, group_results, nside, instrument, args)
        cmb_pytree_list.append(cmb_pytree)
        cl_pytree_list.append(cl_pytree)
        r_pytree_list.append(r_pytree)

    if args.plot_illustrations:
        plot_all_variances(args.titles, cmb_pytree_list)
    if args.plot_all_cmb_recon:
        plot_all_cmb(args.titles, cmb_pytree_list)
    if args.plot_all_spectra:
        plot_all_cl_residuals(args.titles, cl_pytree_list)
        plot_all_r_estimation(args.titles, r_pytree_list)

    plt.show()


if __name__ == "__main__":
    main()
