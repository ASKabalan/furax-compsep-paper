import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["EQX_ON_ERROR"] = "nan"

import sys

import camb
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from furax import HomothetyOperator
from furax.comp_sep import sky_signal
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesQU
from jax_healpy import combine_masks

sys.path.append("../data")
from instruments import get_instrument


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

    return parser.parse_args()


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
    cl_model = r * cl_bb_r1 + cl_bb_lens + cl_noise
    term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term)


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
    r_grid = np.linspace(0, 0.004, 1000)
    logL = np.array(
        [
            log_likelihood(r, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky)
            for r in r_grid
        ]
    )
    L = np.exp(logL - np.max(logL))
    r_best = r_grid[np.argmax(L)]

    rs_pos, L_pos = r_grid[r_grid > r_best], L[r_grid > r_best]
    rs_neg, L_neg = r_grid[r_grid < r_best], L[r_grid < r_best]
    cum_pos = np.cumsum(L_pos) / np.sum(L_pos)
    cum_neg = np.cumsum(L_neg[::-1]) / np.sum(L_neg)

    sigma_pos = rs_pos[np.argmin(np.abs(cum_pos - 0.68))] - r_best if len(rs_pos) > 0 else 0
    sigma_neg = r_best - rs_neg[::-1][np.argmin(np.abs(cum_neg - 0.68))] if len(rs_neg) > 0 else 0
    return r_best, sigma_neg, sigma_pos, r_grid, L


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
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=1024)
    cl_bb_r1_full, cl_bb_total = powers["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_full = cl_bb_total - cl_bb_r1_full

    ell_min, ell_max = 2, nside * 2
    ell_range = np.arange(ell_min, ell_max)
    coeff = ell_range * (ell_range + 1) / (2 * np.pi)
    cl_bb_r1 = cl_bb_r1_full[ell_range] / coeff
    cl_bb_lens = cl_bb_lens_full[ell_range] / coeff
    return ell_range, cl_bb_r1, cl_bb_lens


def compute_w(nu, d, results):
    """
    Apply the linear component separation operator W to the input sky.

    Args:
        nu (np.ndarray): Instrument frequencies.
        d (Stokes): Input sky without CMB.
        results (dict): Fitted spectral parameters.

    Returns:
        StokesQU: Reconstructed CMB map.
    """
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    params = {k: results[k] for k in ["beta_dust", "beta_pl", "temp_dust"]}
    patches = {k: results[k] for k in ["beta_dust_patches", "beta_pl_patches", "temp_dust_patches"]}
    params = jax.tree.map(lambda x: x.mean(axis=0), params)

    def W(p):
        N = HomothetyOperator(1.0, _in_structure=d.structure)
        return sky_signal(
            p, nu, N, d, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patches
        )["cmb"]

    return W(params)


def compute_systematic_res(Wd_cmb, ell_range):
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
    Wn_cl = Wn_cl[2][ell_range]  # shape (lmax)
    return Wn_cl


def compute_total_res(s_hat, true_s, ell_range):
    """
    Compute average BB residual spectrum from multiple noisy realizations.

    Args:
        s_hat (StokesQU): Reconstructed CMB (n_sims, ...)
        true_s (StokesQU): Ground truth CMB map.
        ell_range (np.ndarray): Multipole moments.

    Returns:
        np.ndarray: Residual BB spectrum.
    """
    s_hat = expand_stokes(s_hat)
    true_s = expand_stokes(true_s)
    s_res = s_hat - true_s[np.newaxis, ...]
    # Stack to shape (nsims, 3, npix)
    s_res = np.stack([s_res.i, s_res.q, s_res.u], axis=1)

    cl_list = []
    for i in range(s_res.shape[0]):
        cl = hp.anafast(s_res[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    return np.mean(cl_list, axis=0)  # shape (len(ell_range),)


def compute_cl_obs_bb(s_hat, ell_range):
    """
    Compute the average observed Cl_BB from reconstructed maps.

    Args:
        s_hat (StokesQU): Reconstructed maps (n_sims, ...).
        ell_range (np.ndarray): Multipole moments.

    Returns:
        np.ndarray: Averaged BB spectrum.
    """
    s_hat = expand_stokes(s_hat)
    s_hat = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)
    coeff = ell_range * (ell_range + 1) / (2 * np.pi)

    cl_list = []
    for i in range(s_hat.shape[0]):
        cl = hp.anafast(s_hat[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    return np.mean(cl_list, axis=0) / coeff  # shape (len(ell_range),)


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
    filter_kw = [kw.split("_") for kw in args.runs]
    results_kw = {kw: kw.split("_") for kw in results}

    results_to_plot = []
    for filt_kw in filter_kw:
        group = []
        for result_name, res_kw in results_kw.items():
            if all(kw in res_kw for kw in filt_kw):
                group.append(os.path.join(result_folder, result_name))

        if len(group) > 0:
            results_to_plot.append(group)

    print("Results to plot: ", results_to_plot)
    assert len(args.titles) == len(results_to_plot), "Number of names must match number of results"
    cmb_pytree_list = []
    cl_pytree_list = []
    r_pytree_list = []
    for name , group_results in zip(args.titles, results_to_plot):
        cmb_pytree, cl_pytree, r_pytree = plot_results(name , group_results, nside, instrument)
        cmb_pytree_list.append(cmb_pytree)
        cl_pytree_list.append(cl_pytree)
        r_pytree_list.append(r_pytree)

    plot_all_cmb(args.titles , cmb_pytree_list)


def plot_all_cmb(names , cmb_pytree_list):
    nb_cmb = len(cmb_pytree_list)
    _ = plt.figure(figsize=(8, 4 * nb_cmb))
    for i, cmb_pytree in enumerate(cmb_pytree_list):
        hp.mollview(
            cmb_pytree["cmb"].q - cmb_pytree["recon_mean"].q,
            title=f"Difference (Q) {names[i]}",
            sub=(nb_cmb, 2, i + 1),
            bgcolor=(0.0,) * 4,
        )
        hp.mollview(
            cmb_pytree["cmb"].u - cmb_pytree["recon_mean"].u,
            title=f"Difference (U) {names[i]}",
            sub=(nb_cmb, 2, i + 1 + nb_cmb),
            bgcolor=(0.0,) * 4,
        )
    plt.show()


def plot_cmb_reconsturctions(name ,cmb_stokes, cmb_recon_mean):
    def mse(a, b):
        seen_x = jax.tree.map(lambda x: x[x != hp.UNSEEN], a)
        seen_y = jax.tree.map(lambda x: x[x != hp.UNSEEN], b)
        return jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), seen_x, seen_y)

    mse_cmb = mse(cmb_recon_mean, cmb_stokes)
    cmb_recon_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_recon_mean)
    cmb_input_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_stokes)
    print("======================")
    print(f"MSE CMB: {mse_cmb}")
    print(f"Reconstructed CMB variance: {cmb_recon_var}")
    print(f"Input CMB variance: {cmb_input_var}")
    print("======================")

    _ = plt.figure(figsize=(12, 8))
    hp.mollview(cmb_recon_mean.q, title="Reconstructed CMB (Q)", sub=(2, 3, 1), bgcolor=(0.0,) * 4)
    hp.mollview(cmb_stokes.q, title="Input CMB Map (Q)", sub=(2, 3, 2), bgcolor=(0.0,) * 4)
    hp.mollview(
        cmb_recon_mean.q - cmb_stokes.q,
        title="Difference (Q)",
        sub=(2, 3, 3),
        bgcolor=(0.0,) * 4,
    )
    hp.mollview(cmb_recon_mean.u, title="Reconstructed CMB (U)", sub=(2, 3, 4), bgcolor=(0.0,) * 4)
    hp.mollview(cmb_stokes.u, title="Input CMB Map (U)", sub=(2, 3, 5), bgcolor=(0.0,) * 4)
    hp.mollview(
        cmb_recon_mean.u - cmb_stokes.u,
        title="Difference (U)",
        sub=(2, 3, 6),
        bgcolor=(0.0,) * 4,
    )
    plt.title(f"{name} CMB Reconstruction")
    plt.tight_layout()
    plt.show()


def plot_cl_residuals(name , cl_bb_obs, cl_syst_res, cl_total_res, cl_stat_res, cl_bb_r1, ell_range):
    _ = plt.figure(figsize=(12, 8))

    # --- Power Spectrum Plot ---
    plt.plot(ell_range, cl_bb_obs, label=r"$C_\ell^{\mathrm{obs}}$", color="green")
    plt.plot(ell_range, cl_total_res, label=r"$C_\ell^{\mathrm{res}}$", color="black")
    plt.plot(ell_range, cl_syst_res, label=r"$C_\ell^{\mathrm{syst}}$", color="blue")
    plt.plot(ell_range, cl_stat_res, label=r"$C_\ell^{\mathrm{stat}}$", color="orange")
    plt.plot(ell_range, cl_bb_r1, label=r"$C_\ell^{\mathrm{BB}}(r=1)$", color="red")

    plt.title(f"{name} BB Power Spectra")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [1e-2 $\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_r_estimator(name ,r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals, f_sky):
    _ = plt.figure(figsize=(12, 8))
    # --- Likelihood Plot ---
    likelihood = L_vals / L_vals.max()
    plt.plot(r_grid, likelihood, label="Likelihood", color="purple")
    plt.axvline(r_best, color="black", linestyle="--", label=rf"$\hat{{r}} = {r_best:.2e}$")
    plt.fill_between(
        r_grid,
        0,
        likelihood,
        where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
        color="purple",
        alpha=0.3,
        label=r"$1\sigma$ interval",
    )

    plt.title(f"{name} Likelihood vs $r$")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # plt.savefig(f"{out_folder}/bb_spectra_and_r_likelihood.pdf")
    plt.show()

    print(f"Estimated r: {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


def plot_results(name , filtered_results, nside, instrument):
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

    cmb_recons, cmb_maps, masks, indices_list, w_d_list = [], [], [], [], []

    for folder in filtered_results:
        run_data = dict(np.load(f"{folder}/results.npz"))
        best_params = dict(np.load(f"{folder}/best_params.npz"))
        mask = np.load(f"{folder}/mask.npy")
        indices = jnp.where(mask == 1)[0]

        # Get best run data
        run_data = jax.tree.map(lambda x: x[0], run_data)

        cmb_true = Stokes.from_stokes(Q=best_params["I_CMB"][0], U=best_params["I_CMB"][1])
        fg_map = Stokes.from_stokes(
            Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1]
        )
        cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])
        wd = compute_w(instrument.frequency, fg_map, run_data)

        cmb_recons.append(cmb_recon)
        cmb_maps.append(cmb_true)
        w_d_list.append(wd)
        masks.append(mask)
        indices_list.append(indices)

    full_mask = np.logical_or.reduce(masks)

    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)
    wd = combine_masks(w_d_list, indices_list, nside)

    cmb_recon_mean = jax.tree.map(lambda x: x.mean(axis=0), combined_cmb_recon)
    plot_cmb_reconsturctions(name , cmb_stokes, cmb_recon_mean)

    ell_range, cl_bb_r1, cl_bb_lens = get_camb_templates(nside=64)

    # Compute the systematic residuals Cl_syst = Cl(W(d_no_cmb))
    cl_syst_res = compute_systematic_res(wd, ell_range)
    # Compute the total residuals Cl_res = <CL(s_hat - s_true)>n
    cl_total_res = compute_total_res(combined_cmb_recon, cmb_stokes, ell_range)
    # Compute the statistical residuals Cl_stat = CL_res - CL_syst
    cl_stat_res = cl_total_res - cl_syst_res
    # Compute observed Cl_obs = <CL(s_hat)>
    cl_bb_obs = compute_cl_obs_bb(combined_cmb_recon, ell_range)

    plot_cl_residuals(name , cl_bb_obs, cl_syst_res, cl_total_res, cl_stat_res, cl_bb_r1, ell_range)

    # --- Likelihood Plot ---
    f_sky = full_mask.sum() / len(full_mask)
    r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
        cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_stat_res, f_sky
    )
    plot_r_estimator(name , r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals, f_sky)

    cmb_pytree = {"cmb": cmb_stokes, "recon_mean": cmb_recon_mean}
    cl_pytree = {
        "cl_bb_obs": cl_bb_obs,
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
        "f_sky": f_sky,
    }

    return cmb_pytree, cl_pytree, r_pytree


if __name__ == "__main__":
    main()
