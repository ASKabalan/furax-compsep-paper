import warnings

import camb
import healpy as hp
import numpy as np
from tqdm import tqdm

from ..logging_utils import error, info, success


def _log_likelihood(r, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky):
    """Gaussian pseudo-Cℓ log-likelihood for the tensor-to-scalar ratio."""
    cl_model = r.reshape(-1, 1) * cl_bb_r1 + cl_bb_lens + cl_noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        term = (2 * ell_range + 1) * (cl_obs / cl_model + np.log(cl_model))
    return -0.5 * f_sky * np.sum(term, axis=1)


def _get_camb_templates(nside):
    """Generate BB templates for r=1 and lensing using CAMB.

    Uses the same cosmological parameters as generate_custom_cmb in
    data/generate_maps.py to ensure consistency between map generation
    and template fitting.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    tuple
        (ell_range, cl_bb_r1, cl_bb_lens) where:
        - ell_range: multipole range array
        - cl_bb_r1: BB spectrum template for r=1
        - cl_bb_lens: lensing BB spectrum template
    """
    # Use same cosmology as generate_custom_cmb for consistency
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=1)
    pars.WantTensors = True
    pars.set_for_lmax(1024, lens_potential_accuracy=1)

    results = camb.get_results(pars)
    # Use raw_cl=True to get C_ell directly (not D_ell with ell(ell+1)/(2pi) factor)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=1024, raw_cl=True)
    cl_bb_r1_full, cl_bb_total = powers["tensor"][:, 2], powers["total"][:, 2]
    cl_bb_lens_full = cl_bb_total - cl_bb_r1_full

    ell_min, ell_max = 2, nside * 2 + 2
    ell_range = np.arange(ell_min, ell_max)
    # No need to divide by coeff since raw_cl=True gives C_ell directly
    cl_bb_r1 = cl_bb_r1_full[ell_range]
    cl_bb_lens = cl_bb_lens_full[ell_range]

    return ell_range, cl_bb_r1, cl_bb_lens


def estimate_r(cl, nside, cl_noise, f_sky, is_cl_obs=False, max_point=0.005):
    """Estimate r and 68% uncertainties from a grid-evaluated likelihood.

    Parameters
    ----------
    cl : ndarray
        Input BB power spectrum. Interpretation depends on is_cl_obs:
        - If is_cl_obs=False (default): residual spectrum (lensing will be added)
        - If is_cl_obs=True: observed spectrum (already contains lensing)
    nside : int
        HEALPix resolution for template generation.
    cl_noise : ndarray
        Noise power spectrum (e.g., statistical residuals).
    f_sky : float
        Sky fraction.
    is_cl_obs : bool, optional
        If True, input spectrum is observed (contains lensing).
        If False (default), input is residual and lensing is added internally.

    Returns
    -------
    tuple
        (r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals, ell_range, cl_bb_r1, cl_bb_lens, cl_obs)
    """
    # Generate CAMB templates internally
    ell_range, cl_bb_r1, cl_bb_lens = _get_camb_templates(nside)

    # Compute observed BB spectrum
    if is_cl_obs:
        cl_obs = cl  # Input already observed (contains lensing)
    else:
        cl_obs = cl + cl_bb_lens  # Input is residual, add lensing

    r_grid = np.linspace(-max_point, max_point, 1000)
    logL = _log_likelihood(r_grid, ell_range, cl_obs, cl_bb_r1, cl_bb_lens, cl_noise, f_sky)
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

    return r_best, sigma_neg, sigma_pos, finite_r, L, ell_range, cl_bb_r1, cl_bb_lens, cl_obs


def estimate_r_from_maps(cmb, cmb_hat=None, syst_map=None, nside=None, max_point=0.005):
    """Estimate r from CMB maps with automatic spectrum computation and masking detection.

    This function handles map-based r estimation by:
    1. Expanding QU maps to IQU format
    2. Computing BB power spectrum
    3. Detecting masking and computing f_sky
    4. Optionally computing residual spectra
    5. Calling estimate_r with appropriate parameters

    Parameters
    ----------
    cmb : ndarray
        True CMB map with shape (3, npix) for IQU or (2, npix) for QU.
    cmb_hat : ndarray, optional
        Reconstructed CMB maps from noise realizations,
        shape (n_realizations, 3, npix) or (n_realizations, 2, npix).
    syst_map : ndarray, optional
        Systematic residual map, shape (3, npix) or (2, npix).
    nside : int, optional
        HEALPix resolution. If not provided, inferred from map size.

    Returns
    -------
    tuple
        Same as estimate_r: (r_best, sigma_r_neg, sigma_r_pos, r_grid, L_vals,
                             ell_range, cl_bb_r1, cl_bb_lens, cl_obs)
    """
    # 1. Expand Stokes format (handle QU → IQU)
    if cmb.ndim == 2 and cmb.shape[0] == 2:
        # QU only: expand to IQU by prepending zeros for I
        zeros = np.zeros((1, cmb.shape[1]))
        cmb_expanded = np.vstack([zeros, cmb])
    elif cmb.ndim == 2 and cmb.shape[0] == 3:
        # Already IQU
        cmb_expanded = cmb
    else:
        raise ValueError(f"Invalid CMB map shape: {cmb.shape}. Expected (2, npix) or (3, npix)")

    # 2. Infer nside (if not provided)
    if nside is None:
        npix = cmb_expanded.shape[1]
        nside = hp.npix2nside(npix)

    # 3. Generate CAMB templates
    ell_range, cl_bb_r1, cl_bb_lens = _get_camb_templates(nside)

    # 4. Compute BB spectrum from CMB map
    cl_all = hp.anafast(cmb_expanded)
    cl_cmb = cl_all[2][ell_range]  # Extract BB component

    # 5. Detect masking and compute f_sky
    # Check Q or U component for masking (I might be zero for QU-only input)
    has_mask = np.any(cmb_expanded == hp.UNSEEN)
    if has_mask:
        npix_total = cmb_expanded.shape[1]
        # Count valid pixels in Q component
        npix_valid = np.sum(cmb_expanded[1] != hp.UNSEEN)
        fsky = npix_valid / npix_total
    else:
        fsky = 1.0

    # 6. Determine mode (is_cl_obs)
    is_cl_obs = not has_mask

    info(f"Mask detected: {has_mask}, f_sky = {fsky:.3f}, is_cl_obs = {is_cl_obs}")
    # 7. Expand auxiliary maps (if provided)
    if cmb_hat is not None:
        # cmb_hat: (n_realizations, n_stokes, npix)
        # Expand each realization from QU to IQU if needed
        if cmb_hat.shape[1] == 2:
            zeros = np.zeros((cmb_hat.shape[0], 1, cmb_hat.shape[2]))
            cmb_hat_expanded = np.concatenate([zeros, cmb_hat], axis=1)
        else:
            cmb_hat_expanded = cmb_hat
    else:
        cmb_hat_expanded = None

    if syst_map is not None:
        # syst_map: (n_stokes, npix)
        if syst_map.shape[0] == 2:
            zeros = np.zeros((1, syst_map.shape[1]))
            syst_map_expanded = np.vstack([zeros, syst_map])
        else:
            syst_map_expanded = syst_map
    else:
        syst_map_expanded = None

    # 8. Compute noise spectrum
    if cmb_hat_expanded is not None and syst_map_expanded is not None:
        # Compute spectra directly (compatible with both masked and unmasked)
        # Statistical residuals
        res = cmb_hat_expanded - cmb_expanded[np.newaxis, ...]
        res_stat = res - syst_map_expanded[np.newaxis, ...]

        cl_list = []
        for i in tqdm(range(res_stat.shape[0]), desc="Computing residual spectra"):
            cl = hp.anafast(res_stat[i])
            cl_list.append(cl[2][ell_range])
        cl_noise = np.mean(cl_list, axis=0) / fsky

        # If masked, need total residual for observed spectrum
        if has_mask:
            cl_list_total = []
            for i in tqdm(
                range(cmb_hat_expanded.shape[0]), desc="Computing total residual spectra"
            ):
                res_total = cmb_hat_expanded[i] - cmb_expanded
                cl = hp.anafast(res_total)
                cl_list_total.append(cl[2][ell_range])
            cl_total_res = np.mean(cl_list_total, axis=0) / fsky
            cl_input = cl_total_res
        else:
            cl_input = cl_cmb
    else:
        # No residual analysis: assume perfect reconstruction
        if has_mask:
            raise ValueError(
                "Cannot estimate r from masked maps without residual analysis. "
                "Provide --cmb-hat and --syst arguments."
            )
        cl_noise = np.zeros_like(cl_cmb)
        cl_input = cl_cmb

    # 9. Call estimate_r
    return estimate_r(cl_input, nside, cl_noise, fsky, is_cl_obs=is_cl_obs, max_point=max_point)


def run_estimate(cmb_path, cmb_hat_path, syst_path, fsky, nside, output_path, output_format):
    """Entry point for 'estimate' subcommand.

    Estimates the tensor-to-scalar ratio r from input spectra or maps,
    following the pattern of run_validate.

    Parameters
    ----------
    cmb_path : str
        Path to CMB data (.npy file).
    cmb_hat_path : str or None
        Optional path to reconstructed CMB maps.
    syst_path : str or None
        Optional path to systematic residual map.
    fsky : float or None
        Sky fraction (required for spectrum input).
    nside : int or None
        HEALPix resolution (inferred from map if not provided).
    output_path : str or None
        Optional path to save results as .npz file.
    output_format : str
        Output format for plot: "png", "pdf", or "show".
    """
    from .plotting import plot_r_estimator

    # Load input data
    cmb_data = np.load(cmb_path)
    cmb_hat = np.load(cmb_hat_path) if cmb_hat_path else None
    # Determine if input is spectrum (1D) or map (2D/3D)
    is_spectrum = cmb_data.ndim == 1

    if is_spectrum:
        # Spectrum mode
        if fsky is None:
            error("--fsky is required when input is a power spectrum")
            return
        if nside is None:
            error("--nside is required when input is a power spectrum")
            return

        info(f"Running r estimation from power spectrum (nside={nside}, fsky={fsky})")

        cl_noise = cmb_hat
        cl_total_res = cmb_data
        r_best, sigma_neg, sigma_pos, r_grid, L, ell_range, cl_bb_r1, cl_bb_lens, cl_obs = (
            estimate_r(cl_total_res, nside, cl_noise, fsky, is_cl_obs=True)
        )

    else:
        # Map mode
        if cmb_data.ndim != 2 or cmb_data.shape[0] not in [2, 3]:
            error(
                f"Invalid CMB map shape: {cmb_data.shape}. "
                f"Expected (2, npix) for QU or (3, npix) for IQU"
            )
            return

        info(f"Running r estimation from maps (shape={cmb_data.shape})")

        # Load optional auxiliary maps
        syst_map = np.load(syst_path) if syst_path else None

        # Estimate from maps
        r_best, sigma_neg, sigma_pos, r_grid, L, ell_range, cl_bb_r1, cl_bb_lens, cl_obs = (
            estimate_r_from_maps(cmb_data, cmb_hat, syst_map, nside)
        )

    # Plot results using plot_r_estimator
    name = "r_estimate"
    plot_r_estimator(
        name=name,
        r_best=r_best,
        sigma_r_neg=sigma_neg,
        sigma_r_pos=sigma_pos,
        r_grid=r_grid,
        L_vals=L,
        output_format=output_format,
    )

    success(f"r estimation complete: r = {r_best:+.6f} +{sigma_pos:.6f} -{sigma_neg:.6f}")

    # Save results if requested
    if output_path:
        np.savez(
            output_path,
            r_best=r_best,
            sigma_neg=sigma_neg,
            sigma_pos=sigma_pos,
            r_grid=r_grid,
            likelihood=L,
            ell_range=ell_range,
            cl_bb_r1=cl_bb_r1,
            cl_bb_lens=cl_bb_lens,
            cl_obs=cl_obs,
        )
        success(f"Results saved to: {output_path}")
