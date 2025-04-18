import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["EQX_ON_ERROR"] = "nan"

import os
import sys

import camb
import healpy as hp
import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
from furax import HomothetyOperator
from furax.comp_sep import sky_signal
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesQU
from jax_healpy import combine_masks

sys.path.append("../data")
from instruments import get_instrument


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
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck"],
        help="Instrument to use",
    )

    parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
    )
    return parser.parse_args()


def expand_stokes(stokes_map):
    if isinstance(stokes_map, StokesIQU):
        return stokes_map

    zeros = np.zeros(shape=stokes_map.shape, dtype=stokes_map.dtype)

    if isinstance(stokes_map, StokesI):
        return StokesIQU(stokes_map, zeros, zeros)
    elif isinstance(stokes_map, StokesQU):
        return StokesIQU(zeros, stokes_map.q, stokes_map.u)


def filter_constant_param(input_dict, indx):
    return jax.tree.map(lambda x: x[indx], input_dict)


def sort_results(results, key):
    indices = np.argsort(results[key])
    return jax.tree.map(lambda x: x[indices], results)


# === R ESTIMATION ===


def log_likelihood(r, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    cl_model = r * cl_bb_r1 + cl_bb_lens + cl_noise
    term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term)


def estimate_r(cl_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    r_grid = np.linspace(0, 0.01, 1000)
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


# === CAMB TEMPLATE ===


def get_camb_templates(nside):
    pars = camb.set_params(
        ombh2=0.022,
        omch2=0.12,
        tau=0.054,
        As=2e-9,
        ns=0.965,
        cosmomc_theta=1.04e-2,
        r=1,
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
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    params = {}
    patch_indices = {}
    params["beta_dust"] = results["beta_dust"]
    params["beta_pl"] = results["beta_pl"]
    params["temp_dust"] = results["temp_dust"]
    patch_indices["beta_dust_patches"] = results["beta_dust_patches"]
    patch_indices["beta_pl_patches"] = results["beta_pl_patches"]
    patch_indices["temp_dust_patches"] = results["temp_dust_patches"]

    params = jax.tree.map(lambda x: x.mean(axis=0), params)

    def W(params):
        N = HomothetyOperator(1.0, _in_structure=d.structure)
        Wd = sky_signal(
            params,
            nu,
            N,
            d,
            dust_nu0=dust_nu0,
            synchrotron_nu0=synchrotron_nu0,
            patch_indices=patch_indices,
        )
        return Wd["cmb"]

    Wd_cmb = W(params)  # shape (QU , masked_npix)
    return Wd_cmb


def compute_systematic_res(Wd_cmb, ell_range):
    Wd_cmb = expand_stokes(Wd_cmb)
    Wd_cmb = np.stack([Wd_cmb.i, Wd_cmb.q, Wd_cmb.u], axis=0)  # shape (3 , masked_npix)
    Wn_cl = hp.sphtfunc.anafast(Wd_cmb)
    Wn_cl = Wn_cl[2][ell_range]  # shape (lmax)
    return Wn_cl


def compute_total_res(s_hat, true_s, ell_range):
    # Subtract true map from each noisy realization
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
    s_hat = expand_stokes(s_hat)
    s_hat = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)

    cl_list = []
    for i in range(s_hat.shape[0]):
        cl = hp.anafast(s_hat[i])  # shape (6, lmax+1)
        cl_list.append(cl[2][ell_range])  # BB only

    return np.mean(cl_list, axis=0)  # shape (len(ell_range),)


# === PLOT ===


def main():
    args = parse_args()
    nside = args.nside
    instrument = get_instrument(args.instrument)
    result_folder = "../results/"
    print("Loading data...")
    results = os.listdir(result_folder)
    filter_kw = [kw.split("_") for kw in args.runs]
    results_kw = {kw: kw.split("_") for kw in results}

    results_to_plot = []
    for result_name, res_kw in results_kw.items():
        for filt_kw in filter_kw:
            if all(kw in res_kw for kw in filt_kw):
                results_to_plot.append(result_name)
                break

    print("Results to plot: ", results_to_plot)

    results_to_plot = [f"{result_folder}{res}" for res in results_to_plot]

    PTEP_results = [res for res in results_to_plot if "PTEP" in res]
    # comp_sep_results = [res for res in results_to_plot if "compsep" in res]

    plot_PTEP(PTEP_results, nside, instrument)


def plot_PTEP(PTEP_results, nside, instrument):
    if len(PTEP_results) == 0:
        print("No PTEP results")
        return None, None

    cmb_recons = []
    cmb_maps = []
    masks = []
    indices_list = []
    w_d_list = []

    for res_folder in PTEP_results:
        run_data = dict(np.load(f"{res_folder}/results.npz"))
        best_params = dict(np.load(f"{res_folder}/best_params.npz"))
        cmb_true = best_params["I_CMB"]
        fg_map = best_params["I_D_NOCMB"]
        cmb_recon = run_data["CMB_O"]

        mask = np.load(f"{res_folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)

        cmb_recon = Stokes.from_stokes(Q=cmb_recon[:, 0], U=cmb_recon[:, 1])
        cmb_map_stokes = Stokes.from_stokes(Q=cmb_true[0], U=cmb_true[1])
        fg_map_stokes = Stokes.from_stokes(Q=fg_map[:, 0], U=fg_map[:, 1])

        wd = compute_w(
            instrument.frequency,
            fg_map_stokes,
            run_data,
        )

        cmb_recons.append(cmb_recon)
        cmb_maps.append(cmb_map_stokes)
        w_d_list.append(wd)
        masks.append(mask)
        indices_list.append(indices)

    full_mask = np.zeros_like(masks[0])
    for mask in masks:
        full_mask = np.logical_or(full_mask, mask)

    (full_mask_indices,) = jnp.where(full_mask == 1)
    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside, axis=1)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)
    wd = combine_masks(w_d_list, indices_list, nside)

    ell_range, cl_bb_r1, cl_bb_lens = get_camb_templates(nside=64)

    # Compute the systematic residuals Cl_syst = Cl(W(d_no_cmb))
    cl_syst_res = compute_systematic_res(wd, ell_range)
    # Compute the total residuals Cl_res = <CL(s_hat - s_true)>n
    cl_total_res = compute_total_res(combined_cmb_recon, cmb_stokes, ell_range)
    # Compute the statistical residuals Cl_stat = CL_res - CL_syst
    cl_stat_res = cl_total_res - cl_syst_res
    # Compute observed Cl_obs = <CL(s_hat)>
    cl_bb_obs = compute_cl_obs_bb(combined_cmb_recon, ell_range)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # --- Power Spectrum Plot ---
    axs[0].plot(ell_range, cl_bb_obs, label=r"$C_\ell^{\mathrm{obs}}$", color="green")
    axs[0].plot(ell_range, cl_total_res, label=r"$C_\ell^{\mathrm{res}}$", color="black")
    axs[0].plot(ell_range, cl_syst_res, label=r"$C_\ell^{\mathrm{syst}}$", color="blue")
    axs[0].plot(ell_range, cl_stat_res, label=r"$C_\ell^{\mathrm{stat}}$", color="orange")
    axs[0].plot(ell_range, cl_bb_r1, label=r"$C_\ell^{\mathrm{BB}}(r=1)$", color="red")

    axs[0].set_title("BB Power Spectra")
    axs[0].set_xlabel(r"Multipole $\ell$")
    axs[0].set_ylabel(r"$C_\ell^{BB}$ [1e-2 $\mu K^2$]")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.4)
    axs[0].legend()

    # --- Likelihood Plot ---
    f_sky = full_mask.sum() / len(full_mask)
    r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals = estimate_r(
        cl_bb_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_stat_res, f_sky
    )
    likelihood = L_vals / L_vals.max()
    axs[1].plot(r_grid, likelihood, label="Likelihood", color="purple")
    axs[1].axvline(r_best, color="black", linestyle="--", label=rf"$\hat{{r}} = {r_best:.2e}$")
    axs[1].fill_between(
        r_grid,
        0,
        likelihood,
        where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
        color="purple",
        alpha=0.3,
        label=r"$1\sigma$ interval",
    )

    axs[1].set_title("Likelihood vs $r$")
    axs[1].set_xlabel(r"$r$")
    axs[1].set_ylabel("Relative Likelihood")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    # plt.savefig(f"{out_folder}/bb_spectra_and_r_likelihood.pdf")
    plt.show()

    print(f"Estimated r: {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")


if __name__ == "__main__":
    main()
