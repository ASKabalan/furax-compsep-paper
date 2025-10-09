import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["EQX_ON_ERROR"] = "nan"

import sys
from functools import partial
import os

import camb
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from fgbuster import get_sky
from furax import HomothetyOperator
from furax.obs import (
    negative_log_likelihood,
    sky_signal,
)
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesQU
from jax_grid_search import ProgressBar, optimize
from jax_healpy.clustering import combine_masks
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm import tqdm

sys.path.append("../data")
import itertools

import scienceplots  # noqa: F401
from instruments import get_instrument

# Set the style for the plots
plt.style.use("science")
font_size = 14
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "font.family": "serif",  # or 'Times New Roman' to match LaTeX
        "legend.frameon": True,  # Add boxed legends
    }
)


out_folder = "plots/"

jax.config.update("jax_enable_x64", True)


# ========== Argument Parsing ==========
# ======================================


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
        "-pr",
        "--plot-r-estimation",
        action="store_true",
        help="Plot R estimation for individual runs",
    )
    parser.add_argument(
        "-as",
        "--plot-all-spectra",
        action="store_true",
        help="Plot all spectra of the results",
    )
    parser.add_argument(
        "-ac",
        "--plot-all-cmb-recon",
        action="store_true",
        help="Plot all CMB reconstructions of the results",
    )
    parser.add_argument(
        "-ar",
        "--plot-all-r-estimation",
        action="store_true",
        help="Plot R estimation comparison across all runs",
    )
    parser.add_argument(
        "-a",
        "--plot-all",
        action="store_true",
        help="Plot all results",
    )
    parser.add_argument(
        "-co",
        "--cache-only",
        action="store_true",
        help="Only compute and cache W_D_FG, skip all plotting",
    )
    parser.add_argument(
        "-cr",
        "--compute-residuals",
        type=str,
        choices=["all", "total", "statistical", "systematic", "none"],
        default="none",
        help="Which residuals to compute: all, total, statistical, systematic, or none",
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


def index_run_data(run_data, run_index):
    """
    Index run_data arrays, skipping cached values.

    Args:
        run_data (dict): Dictionary containing run data and cached values.
        run_index (int): Index to extract from non-cached arrays.

    Returns:
        dict: Dictionary with indexed entries (cached values unchanged).
    """

    def should_index(path, value):
        key = path[-1].key if path else None
        if key and (key.startswith("W_D_FG_") or key.startswith("CL_BB_SUM_")):
            return value
        return value[run_index]

    return jax.tree_util.tree_map_with_path(should_index, run_data)


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
# =============================================


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


# ========== CL Computation Functions ==========
# ==============================================


def compute_w(nu, d, results, result_file, run_index=0):
    """
    Apply the linear component separation operator W to the input sky.

    Args:
        nu (np.ndarray): Instrument frequencies.
        d (Stokes): Input sky without CMB.
        results (dict): Fitted spectral parameters.
        result_file (str): Name of the result file.
        run_index (int): Index for caching (default: 0).

    Returns:
        StokesQU: Reconstructed CMB map.
    """
    cache_key = f"W_D_FG_{run_index}"
    if results.get(cache_key) is not None and True:
        print(f"Using {cache_key} from results")
        W = results[cache_key]
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
            max_iter=300,
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
    results_from_file[cache_key] = W_numpy
    atomic_save_results(result_file, results_from_file)
    return W


def compute_systematic_res(Wd_cmb, fsky, ell_range):
    """
    Create the systematic residual map (Wd) and compute its BB spectrum.

    Args:
        Wd_cmb (StokesQU): CMB estimated from foreground-only data (systematic map).
        fsky (float): Observed sky fraction used to debias the spectrum.
        ell_range (np.ndarray): Multipole moments to extract.
        return_map (bool): If True, also return the stacked IQU map array.

    Returns:
        np.ndarray | Tuple[np.ndarray, np.ndarray]:
            BB power spectrum of systematics (len(ell_range),). If ``return_map`` is True,
            also returns the stacked IQU map array with shape (3, npix).
    """
    # Build full IQU map for the systematic residual
    Wd_cmb = expand_stokes(Wd_cmb)
    syst_map = np.stack([Wd_cmb.i, Wd_cmb.q, Wd_cmb.u], axis=0)  # (3, npix)

    # Compute BB and debias by f_sky (maps are not divided by f_sky)
    cl_all = hp.anafast(syst_map)
    cl_bb = cl_all[2][ell_range]
    cl_bb = cl_bb / fsky

    return cl_bb, syst_map


def compute_statistical_res(
    s_hat,
    s_true,
    fsky,
    ell_range,
    s_syst_map: np.ndarray,
):
    """
    Build residual maps and compute the mean BB residual spectrum across simulations.

    Args:
        s_hat (StokesQU): Reconstructed CMB maps with shape (n_sims, ...).
        s_true (StokesQU | np.ndarray): Ground truth CMB map; accepts Stokes or (3, npix) array.
        fsky (float): Observed sky fraction used to debias the spectrum.
        ell_range (np.ndarray): Multipole range to extract.
        s_syst_map (np.ndarray | None): Optional systematic map (Wd) with shape (3, npix).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Mean BB statistical residual spectrum over simulations (len(ell_range),) and
            statistical residual map array with shape (n_sims, 3, npix).
    """
    # Normalize inputs to arrays
    s_hat = expand_stokes(s_hat)
    s_hat_arr = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)  # (n_sims, 3, npix)

    # Total residual maps: res = s_hat - s_true
    res = np.where(s_hat_arr == hp.UNSEEN, hp.UNSEEN, s_hat_arr - s_true[np.newaxis, ...])

    # Statistical residual maps: res_stat = res - s_syst
    s_syst_arr = np.asarray(s_syst_map)  # (3, npix)
    res_stat = np.where(res == hp.UNSEEN, hp.UNSEEN, res - s_syst_arr[np.newaxis, ...])

    # Compute BB spectrum per realisation and average
    cl_list = []
    for i in tqdm(range(res_stat.shape[0]), desc="Computing Statistical BB Spectra"):
        cl = hp.anafast(res_stat[i])  # (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    cl_mean = np.mean(cl_list, axis=0) / fsky

    return cl_mean, res_stat

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
    for i in tqdm(range(res.shape[0]), desc="Computing Residual BB Spectra"):
        cl = hp.anafast(res[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    cl_mean = np.mean(cl_list, axis=0) / fsky  # shape (len(ell_range),)

    return cl_mean, res


def compute_cl_bb_sum(cmb_out, fsky, ell_range):
    """
    Compute BB power spectrum sum from CMB output maps.

    Args:
        cmb_out (Stokes): CMB maps from multiple realizations.
        fsky (float): Sky fraction for debiasing the power spectrum.
        ell_range (np.ndarray): Range of multipole moments to compute.

    Returns:
        np.ndarray: BB power spectrum sum for each realization.
    """
    cmb_out = expand_stokes(cmb_out)
    cmb_out = np.stack([cmb_out.i, cmb_out.q, cmb_out.u], axis=1)

    cl_list = []
    for i in tqdm(range(cmb_out.shape[0]), desc="Computing CL_BB_SUM"):
        cl = hp.anafast(cmb_out[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    CL_BB_SUM = np.sum(cl_list, axis=1) / fsky  # shape (noise realisations,)
    return CL_BB_SUM


def compute_cl_obs_bb(cl_total_res, cl_bb_lens):
    """
    Compute observed BB power spectrum by combining residuals and lensing.

    Args:
        cl_total_res (np.ndarray): Total residual power spectrum.
        cl_bb_lens (np.ndarray): Lensing BB power spectrum.

    Returns:
        np.ndarray: Observed BB power spectrum.
    """
    return cl_total_res + cl_bb_lens


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


# ========== Visualization Functions ==========
# =============================================

# --- Illustration and Analysis Plots ---


def params_to_maps(run_data, previous_mask_size):
    """
    Convert parameter arrays to HEALPix maps for visualization.

    Args:
        run_data (dict): Dictionary containing fitted spectral parameters.
        previous_mask_size (dict): Dictionary tracking cumulative patch sizes.

    Returns:
        Tuple[dict, dict, dict]: (params, patches, updated_mask_size) containing:
            - params: Mean parameter values mapped to patches
            - patches: Normalized patch indices
            - updated_mask_size: Updated cumulative mask sizes
    """
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


def plot_params_patches(name, params, patches, plot_vertical=True):
    """
    Plot parameter maps and their corresponding patch assignments.

    Args:
        name (str): Name for plot titles and output files.
        params (dict): Dictionary containing parameter maps (beta_dust, temp_dust, beta_pl).
        patches (dict): Dictionary containing patch assignment maps.
        plot_vertical (bool): Whether to arrange subplots vertically (default: True).
    """
    with plt.rc_context(
        {
            "font.size": font_size * 1.6,
            "axes.labelsize": font_size * 1.6,
            "xtick.labelsize": font_size * 1.6,
            "ytick.labelsize": font_size * 1.6,
            "legend.fontsize": font_size * 1.6,
            "axes.titlesize": font_size * 1.6,
        }
    ):
        # Params on a figure
        if plot_vertical:
            fig_size = (8, 16)
            subplot_args = (3, 1, lambda i: i + 1)  # 3 rows, 1 column
        else:
            fig_size = (16, 8)
            subplot_args = (1, 3, lambda i: i + 1)  # 1 row, 3 columns

        _ = plt.figure(figsize=fig_size)

        keys = ["beta_dust", "temp_dust", "beta_pl"]
        names = ["$\\beta_d$", "$T_d$", "$\\beta_s$"]

        for i, (key, param_name) in enumerate(zip(keys, names)):
            param_map = params[key]
            hp.mollview(
                param_map,
                title=f"{name} {param_name}",
                sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
                bgcolor=(0.0,) * 4,
                cbar=True,
            )

        plt.tight_layout()
        plt.savefig(f"{out_folder}/params_{name}.pdf", transparent=True, dpi=1200)
        # Create params_dict
        params_dict = {
            "beta_dust": params["beta_dust"],
            "temp_dust": params["temp_dust"],
            "beta_pl": params["beta_pl"],
        }
        np.savez(f"{out_folder}/params_{name}.npz", **params_dict)

        # Patches on a figure
        _ = plt.figure(figsize=fig_size)

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

        # Create patches_dict
        patches_dict = {
            "beta_dust_patches": patches["beta_dust_patches"],
            "temp_dust_patches": patches["temp_dust_patches"],
            "beta_pl_patches": patches["beta_pl_patches"],
        }
        np.savez(f"{out_folder}/patches_{name}.npz", **patches_dict)
        patches = jax.tree.map(shuffle_labels, patches)

        keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
        names = ["$\\beta_d$ Patches", "$T_d$ Patches", "$\\beta_s$ Patches"]

        for i, (key, patch_name) in enumerate(zip(keys, names)):
            patch_map = patches[key]
            hp.mollview(
                patch_map,
                title=f"{name} {patch_name}",
                sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
                bgcolor=(0.0,) * 4,
                cbar=True,
            )
        plt.tight_layout()
        plt.savefig(f"{out_folder}/patches_{name}.pdf", transparent=True, dpi=1200)


def plot_validation_curves(name, updates_history, value_history):
    """
    Plot optimization validation curves showing update norms and NLL history.

    Args:
        name (str): Name for plot title and output file.
        updates_history (array): History of parameter update norms per iteration.
        value_history (array): History of negative log-likelihood values per iteration.
    """
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


# --- Multi-Run Comparison Plots ---


def get_min_variance(cmb_map):
    """
    Select the CMB realization with minimum variance from multiple realizations.

    Args:
        cmb_map (Stokes): CMB maps from multiple realizations.

    Returns:
        Stokes: Single CMB map with the lowest variance.
    """
    seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
    cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
    variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
    variance = sum(jax.tree.leaves(variance))
    argmin = jnp.argmin(variance)
    return jax.tree.map(lambda x: x[argmin], cmb_map)


def plot_all_cmb(names, cmb_pytree_list):
    """
    Plot CMB reconstruction differences for all runs in a comparison plot.

    Args:
        names (list): List of run names for labeling.
        cmb_pytree_list (list): List of CMB data structures containing true and reconstructed maps.
    """
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

    plt.figure(figsize=(10, 3.5 * nb_cmb))

    for i, (name, (diff_q, diff_u)) in enumerate(zip(names, diff_all)):
        # Q map
        hp.mollview(
            diff_q,
            title=rf"Difference (Q) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 1),
            cbar=True,
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )
        # U map
        hp.mollview(
            diff_u,
            title=rf"Difference (U) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 2),
            cbar=True,
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )

    plt.tight_layout()
    name = "_".join(names)
    plt.savefig(f"{out_folder}/cmb_recon_{name}.pdf", transparent=True, dpi=1200)


def plot_all_variances(names, cmb_pytree_list):
    """
    Plot histograms comparing variance distributions across multiple runs.

    Args:
        names (list): List of run names for labeling.
        cmb_pytree_list (list): List of CMB data structures containing reconstruction metrics.
    """

    def get_all_variances(cmb_map):
        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance = sum(jax.tree.leaves(variance))  # shape (100,)
        return variance

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False)

    metrics = {
        "Variance of Reconstructed CMB (Q + U)": [],
        "Negative Log-Likelihood": [],
        r"$\sum C_\ell^{BB}$": [],
    }

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        metrics["Variance of Reconstructed CMB (Q + U)"].append(
            (name, get_all_variances(cmb_pytree["cmb_recon"]))
        )
        metrics["Negative Log-Likelihood"].append((name, np.array(cmb_pytree["nll_summed"])))
        metrics[r"$\sum C_\ell^{BB}$"].append((name, np.array(cmb_pytree["cl_bb_sum"])))

    for ax, (title, entries) in zip(axs, metrics.items()):
        for i, (name, values) in enumerate(entries):
            color = plt.cm.tab10(i % 10)
            label = f"{name}"
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

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize="small", loc="best")

        # Optional: Rotate x-tick labels for better readability if needed
        for label in ax.get_xticklabels():
            label.set_rotation(30)

    axs[-1].set_xlabel("Metric Value", fontsize=12)

    plt.tight_layout(pad=2.0)
    name = "_".join(names)
    plt.savefig(
        f"{out_folder}/metric_distributions_histogram_{name}.pdf", transparent=True, dpi=300
    )


def plot_all_cl_residuals(names, cl_pytree_list):
    """
    Plot BB power spectra comparison for systematic and statistical residuals across all runs.

    Args:
        names (list): List of run names for labeling.
        cl_pytree_list (list): List of power spectrum data structures.
    """
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
        cl_bb_r1 * coeff,
        label=r"$C_\ell^{\mathrm{BB}}(r=1)$",
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
        # Use specific colors based on method name for better distinction
        if "This work" in name or "K-means" in name:
            color = "red"
            linewidth = 2
        elif "Multi-nside" in name:
            if "(1)" in name:
                color = "blue"
            elif "(2)" in name:
                color = "green"
            else:
                color = "blue"  # default for multi-resolution
            linewidth = 1.5
        else:
            color = colors[i % len(colors)]  # fallback to default colors
            linewidth = 1
        # plt.plot(
        #    ell_range,
        #    cl_pytree["cl_bb_obs"] * coeff,
        #    label=rf"{name} $C_\ell^{{\mathrm{{obs}}}}$",
        #    color=color,
        #    linestyle="-",
        # )
        if cl_pytree["cl_total_res"] is not None:
            plt.plot(
            ell_range,
            cl_pytree["cl_total_res"] * coeff,
            label=rf"{name} $C_\ell^{{\mathrm{{res}}}}$",
            color=color,
            linestyle="--",
            )
        if cl_pytree["cl_syst_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_syst_res"] * coeff,
                label=rf"{name} $C_\ell^{{\mathrm{{syst}}}}$",
                color=color,
                linestyle="-",
                linewidth=linewidth,
            )
        if cl_pytree["cl_stat_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_stat_res"] * coeff,
                label=rf"{name} $C_\ell^{{\mathrm{{stat}}}}$",
                color=color,
                linestyle=":",
                linewidth=linewidth,
            )

    plt.title(None)
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    name = "_".join(names)
    plt.savefig(f"{out_folder}/bb_spectra_{name}.pdf", transparent=True, dpi=1200)


def plot_all_systematic_residuals(names, syst_map_list):
    """
    Plot systematic residuals from all runs in one comparison plot.

    Args:
        names (list): List of run names.
        syst_map_list (list): List of systematic residual maps, each with shape (3, npix).
    """
    nb_runs = len(syst_map_list)
    if nb_runs == 0:
        print("No systematic residual maps to plot")
        return

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, syst_map) in enumerate(zip(names, syst_map_list)):
        # Handle UNSEEN pixels for visualization
        syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
        syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

        # Q map
        hp.mollview(
            syst_q,
            title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        # U map
        hp.mollview(
            syst_u,
            title=rf"Systematic Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    plt.tight_layout()
    name = "_".join(names)
    plt.savefig(f"{out_folder}/all_systematic_residuals_{name}.pdf", transparent=True, dpi=1200)


def plot_all_statistical_residuals(names, stat_map_list):
    """
    Plot statistical residuals from all runs in one comparison plot.

    Args:
        names (list): List of run names.
        stat_map_list (list): List of statistical residual maps, each with shape (n_sims, 3, npix).
    """
    nb_runs = len(stat_map_list)
    if nb_runs == 0:
        print("No statistical residual maps to plot")
        return

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, stat_maps) in enumerate(zip(names, stat_map_list)):
        # Use first statistical map to preserve UNSEEN
        stat_map_first = stat_maps[0]  # (3, npix)

        # Handle UNSEEN pixels for visualization
        stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
        stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

        # Q map
        hp.mollview(
            stat_q,
            title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        # U map
        hp.mollview(
            stat_u,
            title=rf"Statistical Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    plt.tight_layout()
    name = "_".join(names)
    plt.savefig(f"{out_folder}/all_statistical_residuals_{name}.pdf", transparent=True, dpi=1200)


def plot_all_r_estimation(names, r_pytree_list):
    """
    Plot likelihood curves for tensor-to-scalar ratio estimation across all runs.

    Args:
        names (list): List of run names for labeling.
        r_pytree_list (list): List of r estimation data structures containing likelihood curves.
    """
    plt.figure(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, r_data) in enumerate(zip(names, r_pytree_list)):
        if r_data["r_best"] is None:
            print(f"WARNING: No r estimation for {name}, skipping plot.")
            continue

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
    name = "_".join(names)
    plt.savefig(f"{out_folder}/r_likelihood_{name}.pdf", transparent=True, dpi=1200)


def _create_r_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list):
    """
    Create a single r vs clusters plot for a specific patch parameter.

    Args:
        patch_name (str): Display name for the patch parameter (e.g., "Beta Dust")
        patch_key (str): Key to access patch data (e.g., "beta_dust_patches")
        names (list): List of run names
        cmb_pytree_list (list): List of CMB data structures
        r_pytree_list (list): List of r estimation data
    """
    # {clusters: {"name": name, "r_best": r_best , "color": color, "sigma_r_neg": sigma_r_neg, "sigma_r_pos": sigma_r_pos}}
    method_dict = {}
    # {name: color}
    colors = {}

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            print(f"WARNING: No r estimation for {name}, skipping plot.")
            continue

        patches = cmb_pytree["patches_map"]
        patch_data = patches[patch_key]
        n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size
        if n_clusters in method_dict:
            existing_r_values = method_dict[n_clusters]["r_best"]
            if r_data["r_best"] > existing_r_values:
                continue

        if name not in colors:
            color = next(color_cycle)
            colors[name] = color

        method_dict[n_clusters] = {
            "name": name,
            "r_best": r_data["r_best"],
            "color": colors[name],
            "sigma_r_neg": r_data["sigma_r_neg"],
            "sigma_r_pos": r_data["sigma_r_pos"],
        }

    plt.figure(figsize=(8, 6))
    labeled_dict = []
    for n_clusters, data in method_dict.items():
        name = data["name"]
        r_best = data["r_best"]
        color = data["color"]
        sigma_r_neg = data["sigma_r_neg"]
        sigma_r_pos = data["sigma_r_pos"]

        if name not in labeled_dict:
            labeled_dict.append(name)
            # Plot error bars for the best-fit r
            plt.errorbar(
                n_clusters,
                r_best,
                yerr=[[sigma_r_neg], [sigma_r_pos]],
                fmt="o",
                label=name,
                color=color,
                capsize=5,
                elinewidth=1,
                markeredgewidth=1,
            )
            plt.scatter(n_clusters, r_best, label=name, color=color)
        else:
            plt.errorbar(
                n_clusters,
                r_best,
                yerr=[[sigma_r_neg], [sigma_r_pos]],
                fmt="o",
                color=color,
                capsize=5,
                elinewidth=1,
                markeredgewidth=1,
            )
            plt.scatter(n_clusters, r_best, color=color)

    plt.xlabel(f"Number of Clusters ({patch_name})")
    plt.ylabel(r"Best-fit $r$")
    plt.title(f"Best-fit $r$ vs. Number of Clusters ({patch_name})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Create filename based on patch parameter
    filename_suffix = patch_key.replace("_patches", "")
    plt.savefig(f"{out_folder}/r_vs_clusters_{filename_suffix}.pdf", transparent=True, dpi=300)
    plt.close()


def plot_r_vs_clusters(names, cmb_pytree_list, r_pytree_list):
    """
    Plot best-fit r values against the number of clusters for all three patch parameters.
    Creates three separate plots saved as individual PDF files.
    """
    # Create three separate plots for each patch parameter
    patch_configs = [
        ("$\\beta_d$", "beta_dust_patches"),
        ("$T_d$", "temp_dust_patches"),
        ("$\\beta_s$", "beta_pl_patches"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_r_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list)


# --- Single Run Plots ---


def plot_systematic_residual_maps(name, syst_map):
    """
    Plot systematic residual maps.

    Args:
        name (str): Name for the plot title and file.
        syst_map (np.ndarray): Systematic residual map with shape (3, npix).
    """
    # Handle UNSEEN pixels for visualization
    syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
    syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

    plt.figure(figsize=(12, 6))

    # Systematic residual Q
    hp.mollview(
        syst_q,
        title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    # Systematic residual U
    hp.mollview(
        syst_u,
        title=rf"Systematic Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/systematic_residual_maps_{name}.pdf", transparent=True, dpi=1200)


def plot_statistical_residual_maps(name, stat_maps):
    """
    Plot statistical residual maps.

    Args:
        name (str): Name for the plot title and file.
        stat_maps (np.ndarray): Statistical residual maps with shape (n_sims, 3, npix).
    """
    # Use the first statistical map (to preserve UNSEEN)
    stat_map_first = stat_maps[0]  # (3, npix)

    # Handle UNSEEN pixels for visualization
    stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
    stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

    plt.figure(figsize=(12, 6))

    # Statistical residual Q
    hp.mollview(
        stat_q,
        title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    # Statistical residual U
    hp.mollview(
        stat_u,
        title=rf"Statistical Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/statistical_residual_maps_{name}.pdf", transparent=True, dpi=1200)


def plot_cmb_reconstructions(name, cmb_stokes, cmb_recon):
    """
    Plot CMB reconstruction results with MSE analysis.

    Args:
        name (str): Name for plot title and output file.
        cmb_stokes (Stokes): True CMB input maps.
        cmb_recon (Stokes): Reconstructed CMB maps from component separation.
    """

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

    _ = plt.figure(figsize=(12, 12))
    hp.mollview(
        cmb_recon_min.q, title=r"Reconstructed CMB (Q) [$\mu$K]", sub=(3, 3, 1), bgcolor=(0,) * 4
    )
    hp.mollview(cmb_stokes.q, title=r"Input CMB Map (Q) [$\mu$K]", sub=(3, 3, 2), bgcolor=(0,) * 4)
    hp.mollview(
        diff_q,
        title=r"Difference (Q) [$\mu$K]",
        sub=(3, 3, 3),
        cbar=True,
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        cmb_recon_min.u, title=r"Reconstructed CMB (U) [$\mu$K]", sub=(3, 3, 4), bgcolor=(0,) * 4
    )
    hp.mollview(cmb_stokes.u, title=r"Input CMB Map (U) [$\mu$K]", sub=(3, 3, 5), bgcolor=(0,) * 4)
    hp.mollview(
        diff_u,
        title=r"Difference (U) [$\mu$K]",
        sub=(3, 3, 6),
        cbar=True,
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
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
    """
    Plot BB power spectrum residuals for a single run.

    Args:
        name (str): Name for plot title and output file.
        cl_bb_obs (np.ndarray): Observed BB power spectrum.
        cl_syst_res (np.ndarray): Systematic residual power spectrum.
        cl_total_res (np.ndarray): Total residual power spectrum.
        cl_stat_res (np.ndarray): Statistical residual power spectrum.
        cl_bb_r1 (np.ndarray): BB template for r=1.
        cl_bb_r0 (np.ndarray): BB template for r=0.
        cl_bb_lens (np.ndarray): Lensing BB power spectrum.
        cl_true (np.ndarray): True CMB BB power spectrum.
        ell_range (np.ndarray): Multipole moment range.
    """
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
    plt.ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]")
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
    """
    Plot likelihood curve for tensor-to-scalar ratio estimation.

    Args:
        name (str): Name for plot title and output file.
        r_best (float): Best-fit value of r.
        sigma_r_neg (float): Lower uncertainty bound.
        sigma_r_pos (float): Upper uncertainty bound.
        r_grid (np.ndarray): Grid of r values for likelihood evaluation.
        L_vals (np.ndarray): Likelihood values corresponding to r_grid.
    """
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
    plt.savefig(f"{out_folder}/r_likelihood_{name}.pdf", transparent=True, dpi=1200)

    # Print
    print(f"Estimated r (Reconstructed): {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


# ========== Caching Utilities ==========
# ======================================


def atomic_save_results(result_file, results_dict):
    """
    Atomically save results with backup protection.

    Args:
        result_file (str): Path to the results file.
        results_dict (dict): Dictionary to save.
    """
    # Create backup of existing file
    if os.path.exists(result_file):
        backup_file = result_file.replace('.npz', '.bk.npz')
        os.rename(result_file, backup_file)

    # Write to temporary file first
    temp_file = result_file.replace('.npz', '.tmp.npz')
    np.savez(temp_file, **results_dict)

    # Atomic rename to final location
    os.rename(temp_file, result_file)


def check_cache_keys_exist(result_file, run_index):
    """
    Check if W_D_FG cache key exists for given run_index.

    Args:
        result_file (str): Path to results file
        run_index (int): Run index to check

    Returns:
        bool: True if cache key exists, False otherwise
    """
    try:
        with np.load(result_file) as f:
            w_key = f"W_D_FG_{run_index}"
            return w_key in f.keys()
    except (OSError, FileNotFoundError):
        return False


def load_run_data_for_cache(folder, nside, instrument, run_index=0):
    """
    Load minimal data needed for caching W_D_FG.

    Args:
        folder (str): Path to result folder.
        nside (int): HEALPix resolution.
        instrument: Instrument object.
        run_index (int): Index to select from run_data arrays (default: 0).

    Returns:
        Tuple: (run_data, best_params, mask, indices, f_sky, cmb_recon, fg_map)
        Returns None if index is out of bounds.
    """
    run_data = dict(np.load(f"{folder}/results.npz"))
    best_params = dict(np.load(f"{folder}/best_params.npz"))
    mask = np.load(f"{folder}/mask.npy")
    (indices,) = jnp.where(mask == 1)
    f_sky = mask.sum() / len(mask)

    first_key = next(iter(run_data.keys()))
    max_index = len(run_data[first_key]) - 1

    if run_index > max_index:
        print(
            f"WARNING: Index {run_index} out of bounds (max: {max_index}) for folder {folder}. Skipping."
        )
        return None

    run_data = index_run_data(run_data, run_index)

    #cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
    fg_map = Stokes.from_stokes(Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1])
    cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])

    return run_data, best_params, mask, indices, f_sky, cmb_recon, fg_map


def cache_expensive_computations(name, filtered_results, nside, instrument, run_index=0):
    """
    Compute and cache expensive computations (W_D_FG) for given results.

    Args:
        name (str): Name of the run for progress tracking.
        filtered_results (list[str]): List of result directories to process.
        nside (int): HEALPix resolution parameter.
        instrument: Instrument object with frequency specifications.
        run_index (int): Index to select from run_data arrays (default: 0).
    """
    if len(filtered_results) == 0:
        print(f"No results found for {name}")
        return

    print(f"Caching expensive computations for {name} (index={run_index})...")

    for i, folder in enumerate(filtered_results):
        print(f"  Processing folder {i + 1}/{len(filtered_results)}: {folder}")

        if check_cache_keys_exist(f"{folder}/results.npz", run_index):
            print(f"    Cache already exists for index {run_index}, skipping...")
            continue

        try:
            result = load_run_data_for_cache(folder, nside, instrument, run_index)
            if result is None:
                continue

            run_data, best_params, mask, indices, f_sky, cmb_recon, fg_map = result

            print("    Computing/caching W_D_FG...")
            _ = compute_w(
                instrument.frequency,
                fg_map,
                run_data,
                result_file=f"{folder}/results.npz",
                run_index=run_index,
            )

            print(f"    ✓ Completed folder {folder}")

        except Exception as e:
            print(f"    ✗ Error processing folder {folder}: {e}")
            continue

    print(f"✓ Finished caching for {name}")


def plot_results(name, filtered_results, nside, instrument, args, run_index=0):
    """
    Load, combine, and analyze component separation results with comprehensive plotting.

    Args:
        name (str): Name of the run for output files and titles.
        filtered_results (list[str]): List of result directories to process.
        nside (int): HEALPix resolution parameter.
        instrument (Instrument): Instrument object with frequency specifications.
        args: Parsed command-line arguments controlling plot generation.
        run_index (int): Index to select from run_data arrays (default: 0).

    Returns:
        Tuple[dict, dict, dict, dict]: (cmb_pytree, cl_pytree, r_pytree, residual_pytree)
            containing processed results for further analysis.
    """
    if len(filtered_results) == 0:
        print("No results")
        return

    cmb_recons, cmb_maps, masks, NLLs = [], [], [], []
    indices_list, w_d_list = [], []
    params_list, patches_list = [], []
    updates_history, value_history = [], []

    needs_residual_maps = args.plot_cmb_recon or args.plot_all
    needs_residual_spectra = args.plot_cl_spectra or args.plot_all_spectra or args.plot_all
    needs_r_estimation = (
        args.plot_r_estimation
        or args.plot_all_r_estimation
        or args.plot_illustrations
        or args.plot_all
    )

    compute_syst = args.compute_residuals in ["all", "systematic"]
    compute_stat = args.compute_residuals in ["all", "statistical"]
    compute_total = args.compute_residuals in ["all", "total"]


    if needs_residual_spectra or needs_residual_maps:
        compute_syst = True
        compute_stat = True
    if needs_r_estimation:
        compute_total = True
        compute_stat = True
        compute_syst = True

    needs_camb = (
        args.plot_cl_spectra
        or args.plot_all_spectra
        or args.plot_r_estimation
        or args.plot_all_r_estimation
        or args.plot_illustrations
        or args.plot_all
    )
    if needs_camb:
        ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, _ = get_camb_templates(nside=64)
    else:
        ell_range = cl_bb_r1 = cl_bb_r0 = cl_bb_lens = None

    previous_mask_size = {
        "beta_dust_patches": 0,
        "temp_dust_patches": 0,
        "beta_pl_patches": 0,
    }

    for folder in filtered_results:
        print("--------------------------------------------------")
        print(f"Processing folder: {folder}")
        print("--------------------------------------------------")
        run_data = dict(np.load(f"{folder}/results.npz"))
        best_params = dict(np.load(f"{folder}/best_params.npz"))
        mask = np.load(f"{folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)
        f_sky = mask.sum() / len(mask)

        first_key = next(iter(run_data.keys()))
        max_index = len(run_data[first_key]) - 1

        if run_index > max_index:
            print(
                f"WARNING: Index {run_index} out of bounds (max: {max_index}) for folder {folder}. Skipping."
            )
            continue

        run_data = index_run_data(run_data, run_index)
        NLL = run_data["NLL"]

        cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
        fg_map = Stokes.from_stokes(
            Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
        )
        cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])

        if compute_syst:
            wd = compute_w(
                instrument.frequency,
                fg_map,
                run_data,
                result_file=f"{folder}/results.npz",
                run_index=run_index,
            )
        else:
            wd = None

        if args.plot_illustrations:
            params, patches, previous_mask_size = params_to_maps(run_data, previous_mask_size)
            params_list.append(params)
            patches_list.append(patches)

        if args.plot_validation_curves:
            updates_history.append(run_data["update_history"][..., 0])
            value_history.append(run_data["update_history"][..., 1])

        cmb_recons.append(cmb_recon)
        cmb_maps.append(cmb_true)
        if wd is not None:
            w_d_list.append(wd)
        masks.append(mask)
        indices_list.append(indices)
        NLLs.append(NLL)

    if len(masks) == 0:
        print(
            f"WARNING: No valid data found for '{name}' with index {run_index}. Skipping this run."
        )
        return None

    full_mask = np.logical_or.reduce(masks)

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)

    if compute_syst and len(w_d_list) > 0:
        wd = combine_masks(w_d_list, indices_list, nside)
    else:
        wd = None

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

    needs_sky = compute_syst or compute_stat or compute_total
    if needs_sky:
        s_true = get_sky(64, "c1d1s1").components[0].map.value
    else:
        s_true = None

    if args.plot_cmb_recon:
        plot_cmb_reconstructions(name, cmb_stokes, combined_cmb_recon)

    f_sky = full_mask.sum() / len(full_mask)

    cl_syst_res, syst_map, cl_stat_res, stat_maps = None, None, None, None

    if compute_syst and wd is not None:
        cl_syst_res, syst_map = compute_systematic_res(wd, f_sky, ell_range)
        print(f"maximum cl_syst_res: {np.max(cl_syst_res)}")

    print(f"compute_total: {compute_total}, needs_r_estimation: {needs_r_estimation} s_true: {s_true is not None}")

    if compute_stat and compute_syst and syst_map is not None:
        print("Computing statistical residuals...")
        cl_stat_res, stat_maps = compute_statistical_res(
            combined_cmb_recon, s_true, f_sky, ell_range, syst_map
        )

    if compute_total:
        if cl_syst_res is not None and cl_stat_res is not None:
            print("Computing total residuals (as sum of syst + stat)...")
            cl_total_res = cl_syst_res + cl_stat_res
        else:
            print("Computing total residuals (direct computation)...")
            cl_total_res, _ = compute_total_res(combined_cmb_recon, s_true, f_sky, ell_range)
    else:
        cl_total_res = None

    if (args.plot_cmb_recon or args.plot_all):
        if syst_map is not None:
            plot_systematic_residual_maps(name, syst_map)
        if stat_maps is not None:
            plot_statistical_residual_maps(name, stat_maps)

    if args.plot_cl_spectra and ell_range is not None and s_true is not None:
        cl_true = compute_cl_true_bb(s_true, ell_range)
    else:
        cl_true = None

    if compute_total and ell_range is not None:
        cl_bb_obs = compute_cl_obs_bb(cl_total_res, cl_bb_lens)
    else:
        cl_bb_obs = None

    if args.plot_illustrations or args.plot_all:
        cl_bb_sum = compute_cl_bb_sum(combined_cmb_recon, f_sky, ell_range)
    else:
        cl_bb_sum = None

    if args.plot_cl_spectra and cl_bb_obs is not None:
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

    if compute_total and needs_r_estimation and cl_bb_obs is not None:
        print(f"Estimating r for {name}...")
        stat_res_for_r = cl_stat_res if cl_stat_res is not None else np.zeros_like(ell_range)
        r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
            cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, stat_res_for_r, f_sky
        )
    else:
        r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = None, None, None, None, None

    if args.plot_r_estimation and r_best is not None:
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
        "cl_total_res": cl_total_res if compute_total else None,
        "cl_stat_res": cl_stat_res,
    }
    r_pytree = {
        "r_best": r_best,
        "sigma_r_neg": sigma_r_neg,
        "sigma_r_pos": sigma_r_pos,
        "r_grid": r_grid,
        "L_vals": L_vals,
    }
    residual_pytree = {
        "syst_map": syst_map,
        "stat_maps": stat_maps,
    }

    return cmb_pytree, cl_pytree, r_pytree, residual_pytree


# ========== Run Management Utilities ==========
# =============================================


def parse_run_spec(run_spec):
    """
    Parse a run specification like 'NAME', 'NAME,5', or 'NAME,0-15'.

    Returns:
        tuple: (filter_string, index_spec) where index_spec is either:
            - int (single index)
            - tuple (start, end) for range
            - None (default to 0)
    """
    if "," not in run_spec:
        return run_spec, None

    filter_part, index_part = run_spec.rsplit(",", 1)
    index_part = index_part.strip()

    if "-" in index_part:
        start, end = index_part.split("-", 1)
        return filter_part, (int(start.strip()), int(end.strip()))
    else:
        return filter_part, int(index_part)


def parse_filter_kw(kw_string):
    """
    Parse filter keyword string with OR/AND logic for result filtering.

    Args:
        kw_string (str): Filter string like 'A_(B|C)_D' where parentheses indicate OR options.

    Returns:
        list[set]: List of sets representing AND-of-OR filter groups.
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
    Check if name parts satisfy the AND-of-OR filter criteria.

    Args:
        name_parts (list): List of name components to check.
        filter_groups (list[set]): AND-of-OR filter groups from parse_filter_kw.

    Returns:
        bool: True if name_parts matches all filter groups.
    """
    return all(any(option in name_parts for option in group) for group in filter_groups)


def expand_run_specs(run_specs, titles):
    """
    Expand run specifications with ranges into groups of (filter, index, title) tuples.

    Args:
        run_specs (list): List of run specification strings.
        titles (list): List of title strings corresponding to run_specs.

    Returns:
        list: List of groups, where each group is a list of tuples (filter_string, run_index, title_string).
              Single runs are wrapped in a list of length 1.
    """
    expanded = []

    for run_spec, base_title in zip(run_specs, titles):
        filter_str, index_spec = parse_run_spec(run_spec)

        if index_spec is None:
            expanded.append([(filter_str, 0, base_title)])
        elif isinstance(index_spec, int):
            expanded.append([(filter_str, index_spec, base_title)])
        elif isinstance(index_spec, tuple):
            start, end = index_spec
            group = []
            for idx in range(start, end + 1):
                title = f"{base_title} ({idx})"
                group.append((filter_str, idx, title))
            expanded.append(group)
        else:
            raise ValueError(f"Unknown index specification: {index_spec}")

    return expanded


# ========== Main Execution ==========
# ====================================


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
        args.plot_r_estimation = True
        args.plot_all_r_estimation = True
        args.plot_validation_curves = True
        args.plot_illustrations = True

    expanded_run_groups = expand_run_specs(args.runs, args.titles)

    results_to_plot = []
    titles_to_plot = []
    indices_to_plot = []

    for run_group in expanded_run_groups:
        for filter_expr, run_index, title in run_group:
            filter_groups = parse_filter_kw(filter_expr)
            group = []
            for result_name, res_kw in results_kw.items():
                if matches_filter(res_kw, filter_groups):
                    group.append(os.path.join(result_folder, result_name))
            if group:
                results_to_plot.append(group)
                titles_to_plot.append(title)
                indices_to_plot.append(run_index)

    print("Results to plot: ", results_to_plot)
    print("Titles: ", titles_to_plot)
    print("Indices: ", indices_to_plot)

    if args.cache_only:
        print("=" * 60)
        print("CACHE-ONLY MODE: Computing and caching W_D_FG only")
        print("=" * 60)

        for name, group_results, run_index in zip(titles_to_plot, results_to_plot, indices_to_plot):
            cache_expensive_computations(name, group_results, nside, instrument, run_index)

        print("=" * 60)
        print("✓ Cache-only mode completed successfully!")
        print("✓ W_D_FG has been cached for all runs")
        print("✓ You can now run plotting commands to use the cached values")
        print("=" * 60)
        return

    cmb_pytree_list = []
    cl_pytree_list = []
    r_pytree_list = []
    syst_map_list = []
    stat_map_list = []
    valid_titles = []
    for name, group_results, run_index in zip(titles_to_plot, results_to_plot, indices_to_plot):
        result = plot_results(name, group_results, nside, instrument, args, run_index)
        if result is None:
            continue
        cmb_pytree, cl_pytree, r_pytree, residual_pytree = result
        cmb_pytree_list.append(cmb_pytree)
        cl_pytree_list.append(cl_pytree)
        r_pytree_list.append(r_pytree)
        if residual_pytree["syst_map"] is not None:
            syst_map_list.append(residual_pytree["syst_map"])
        if residual_pytree["stat_maps"] is not None:
            stat_map_list.append(residual_pytree["stat_maps"])
        valid_titles.append(name)
        plt.close("all")

    if args.plot_illustrations:
        plot_r_vs_clusters(valid_titles, cmb_pytree_list, r_pytree_list)
        plot_all_variances(valid_titles, cmb_pytree_list)
        plt.close("all")
    if args.plot_all_cmb_recon:
        plot_all_cmb(valid_titles, cmb_pytree_list)
        plt.close("all")
    if args.plot_all_spectra:
        plot_all_cl_residuals(valid_titles, cl_pytree_list)
        if len(syst_map_list) > 0:
            plot_all_systematic_residuals(valid_titles, syst_map_list)
        if len(stat_map_list) > 0:
            plot_all_statistical_residuals(valid_titles, stat_map_list)
        plt.close("all")
    if args.plot_all_r_estimation:
        plot_all_r_estimation(valid_titles, r_pytree_list)
        plt.close("all")


if __name__ == "__main__":
    main()
