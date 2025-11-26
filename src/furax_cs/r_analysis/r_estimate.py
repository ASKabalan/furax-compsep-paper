import warnings

import camb
import numpy as np


def log_likelihood(r, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    """Gaussian pseudo-Cℓ log-likelihood for the tensor-to-scalar ratio."""
    cl_model = r.reshape(-1, 1) * cl_bb_r1 + cl_bb_lens + cl_noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term, axis=1)


def estimate_r(cl_obs, ell_range, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    """Estimate r and 68% uncertainties from a grid-evaluated likelihood."""
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
    """Generate BB templates for r=1 and lensing using CAMB."""
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
    pars_0 = camb.set_params(
        ombh2=0.022,
        omch2=0.12,
        tau=0.054,
        As=2e-9,
        ns=0.965,
        cosmomc_theta=1.04e-2,
        r=0,
        DoLensing=True,
        WantTensors=True,
        Want_CMB_lensing=True,
        lmax=1024,
    )
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=1024)
    cl_bb_r1_full, cl_bb_total = powers["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_full = cl_bb_total - cl_bb_r1_full

    ell_min, ell_max = 2, nside * 2 + 2
    ell_range = np.arange(ell_min, ell_max)
    coeff = ell_range * (ell_range + 1) / (2 * np.pi)
    cl_bb_r1 = cl_bb_r1_full[ell_range] / coeff
    cl_bb_lens = cl_bb_lens_full[ell_range] / coeff

    results_0 = camb.get_results(pars_0)
    powers_0 = results_0.get_cmb_power_spectra(pars_0, CMB_unit="muK", lmax=1024)
    cl_bb_r0_full, cl_bb_total = powers_0["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_r0_full = cl_bb_total - cl_bb_r0_full

    cl_bb_r0 = cl_bb_r0_full[ell_range] / coeff
    cl_bb_r0_lens = cl_bb_lens_r0_full[ell_range] / coeff

    return ell_range, cl_bb_r1, cl_bb_r0, cl_bb_lens, cl_bb_r0_lens
