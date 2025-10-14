import healpy as hp
import numpy as np
from tqdm import tqdm

from .utils import expand_stokes


def compute_systematic_res(Wd_cmb, fsky, ell_range):
    """Compute systematic residual BB power and map.

    Parameters
    ----------
    Wd_cmb : StokesI | StokesQU | StokesIQU
        Foreground-only CMB reconstruction, i.e. W * d_fg.
    fsky : float
        Observed sky fraction used to debias the spectra.
    ell_range : ndarray
        Multipole array over which the spectrum is evaluated.

    Returns
    -------
    tuple
        (C_ell^syst, syst_map) where the spectrum is rescaled by f_sky and the
        map is a (3, Npix) array containing I, Q, U residuals.
    """
    Wd_cmb = expand_stokes(Wd_cmb)
    syst_map = np.stack([Wd_cmb.i, Wd_cmb.q, Wd_cmb.u], axis=0)

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
    """Compute statistical residuals after subtracting systematic leakage.

    Parameters
    ----------
    s_hat : StokesI | StokesQU | StokesIQU
        Reconstructed CMB map for all noise realizations.
    s_true : ndarray
        True input CMB map (I, Q, U stacked along axis=0).
    fsky : float
        Observed sky fraction.
    ell_range : ndarray
        Multipole range for spectral estimation.
    s_syst_map : ndarray
        Systematic residual map returned by :func:`compute_systematic_res`.

    Returns
    -------
    tuple
        (C_ell^stat, stat_maps) with averaged BB spectrum and residual maps per
        realization.
    """
    s_hat = expand_stokes(s_hat)
    s_hat_arr = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)

    res = np.where(s_hat_arr == hp.UNSEEN, hp.UNSEEN, s_hat_arr - s_true[np.newaxis, ...])

    s_syst_arr = np.asarray(s_syst_map)
    res_stat = np.where(res == hp.UNSEEN, hp.UNSEEN, res - s_syst_arr[np.newaxis, ...])

    cl_list = []
    for i in tqdm(range(res_stat.shape[0]), desc="Computing Statistical BB Spectra"):
        cl = hp.anafast(res_stat[i])
        cl_list.append(cl[2][ell_range])

    cl_mean = np.mean(cl_list, axis=0) / fsky

    return cl_mean, res_stat


def compute_total_res(s_hat, s_true, fsky, ell_range):
    """Compute total residual BB spectrum without separating components.

    Parameters
    ----------
    s_hat : StokesI | StokesQU | StokesIQU
        Reconstructed CMB map for all noise realizations.
    s_true : ndarray
        True input CMB map (I, Q, U).
    fsky : float
        Observed sky fraction.
    ell_range : ndarray
        Multipole range for spectral estimation.

    Returns
    -------
    tuple
        (C_ell^res, residual_maps) where residual_maps has shape
        (n_realizations, 3, Npix).
    """
    s_hat = expand_stokes(s_hat)
    s_hat = np.stack([s_hat.i, s_hat.q, s_hat.u], axis=1)

    res = np.where(s_hat == hp.UNSEEN, hp.UNSEEN, s_hat - s_true[np.newaxis, ...])
    cl_list = []
    for i in tqdm(range(res.shape[0]), desc="Computing Residual BB Spectra"):
        cl = hp.anafast(res[i])
        cl_list.append(cl[2][ell_range])

    cl_mean = np.mean(cl_list, axis=0) / fsky

    return cl_mean, res


def compute_cl_bb_sum(cmb_out, fsky, ell_range):
    """Compute ∑ C_ell^{BB} across realizations for the recovered CMB."""
    cmb_out = expand_stokes(cmb_out)
    cmb_out = np.stack([cmb_out.i, cmb_out.q, cmb_out.u], axis=1)

    cl_list = []
    for i in tqdm(range(cmb_out.shape[0]), desc="Computing CL_BB_SUM"):
        cl = hp.anafast(cmb_out[i])
        cl_list.append(cl[2][ell_range])

    CL_BB_SUM = np.sum(cl_list, axis=1) / fsky
    return CL_BB_SUM


def compute_cl_obs_bb(cl_total_res, cl_bb_lens):
    """Combine residual and lensing spectra to form observed BB power."""
    return cl_total_res + cl_bb_lens


def compute_cl_true_bb(s, ell_range):
    """Compute the true sky BB spectrum over the requested ell range."""
    cl = hp.anafast(s)

    return cl[2][ell_range]
