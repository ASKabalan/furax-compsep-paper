# FGBUSTER IMPORTS

import argparse
import os
import pickle
from pathlib import Path

import healpy as hp
import numpy as np
from fgbuster import (
    get_instrument,
    get_observation,
    get_sky,
)
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    SynchrotronOperator,
)


def save_to_cache(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Generate and cache frequency maps for component separation.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise : bool, optional
        Whether to include noise in simulations (default: False).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string (e.g., "c1d0s0" for CMB only) (default: "c1d0s0").

    Returns
    -------
    tuple
        (frequencies, freq_maps) where frequencies are in GHz and freq_maps
        have shape (n_freq, 3, n_pix) for Stokes I, Q, U.
    """
    instrument = get_instrument(instrument_name)
    # Define cache file path
    cache_dir = "freq_maps_cache"
    os.makedirs(cache_dir, exist_ok=True)
    noise_str = "noise" if noise else "no_noise"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists, load if it does, otherwise create and save it
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache and noise {noise}.")
    else:
        # Generate freq_maps if not already cached
        freq_maps = get_observation(instrument, sky, nside=nside, noise=noise)

        # Save freq_maps to the cache
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        print(f"Generated and saved freq_maps for nside {nside}.")
    return instrument["frequency"].values, freq_maps


def load_from_cache(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Load cached frequency maps from disk.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise : bool, optional
        Whether maps include noise (default: False).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string (default: "c1d0s0").

    Returns
    -------
    tuple
        (frequencies, freq_maps) loaded from cache.

    Raises
    ------
    FileNotFoundError
        If cache file does not exist.
    """
    # Define cache file path
    instrument = get_instrument(instrument_name)
    noise_str = "noise" if noise else "no_noise"
    cache_dir = "freq_maps_cache"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        raise FileNotFoundError(
            f"Cache file for freq_maps with nside {nside} not found.\n"
            f"Please generate it first by calling `generate_maps({nside})`."
        )

    return instrument["frequency"].values, freq_maps


def strip_cmb_tag(sky_string):
    """Removes the 'cX' tag from the sky string."""
    tags = [sky_string[i : i + 2] for i in range(0, len(sky_string), 2)]
    fg_tags = [tag for tag in tags if not tag.startswith("c")]
    return "".join(fg_tags)


def save_fg_map(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Generate and cache foreground-only frequency maps (CMB excluded).

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise : bool, optional
        Whether to include noise (default: False).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string, CMB component automatically removed (default: "c1d0s0").

    Returns
    -------
    tuple
        (frequencies, freq_maps) containing only foreground contributions.
    """
    print(f"Generating fg map for nside {nside}, noise {noise}, instrument {instrument_name}")
    stripped_sky = strip_cmb_tag(sky)
    return save_to_cache(nside, noise=noise, instrument_name=instrument_name, sky=stripped_sky)


def load_fg_map(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Load cached foreground-only frequency maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise : bool, optional
        Whether maps include noise (default: False).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string, CMB automatically excluded (default: "c1d0s0").

    Returns
    -------
    tuple
        (frequencies, freq_maps) containing only foreground contributions.
    """
    stripped_sky = strip_cmb_tag(sky)
    return load_from_cache(nside, noise=noise, instrument_name=instrument_name, sky=stripped_sky)


def save_cmb_map(nside, sky="c1d0s0"):
    """Generate and cache CMB-only maps for template generation.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    sky : str, optional
        Sky model preset string (default: "c1d0s0").

    Returns
    -------
    ndarray
        CMB map with shape (3, n_pix) for Stokes I, Q, U, or zeros if no CMB.
    """
    print(f"Generating CMB map for nside {nside}, sky {sky}")
    # Define cache file path
    cache_dir = "freq_maps_cache"
    os.makedirs(cache_dir, exist_ok=True)
    preset_strings = [sky[i : i + 2] for i in range(0, len(sky), 2)]
    cmb_template = None
    for strr in preset_strings:
        if strr.startswith("c"):
            cmb_template = strr
            break

    if cmb_template is None:
        npix = 12 * nside**2
        return np.zeros((3, npix))
    else:
        cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{cmb_template}.pkl")
        sky_obj = get_sky(nside, sky)
        freq_maps = sky_obj.components[0].map.to_value()
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        print(f"Generated and saved freq_maps for nside {nside} and for tag {cmb_template}.")

        return freq_maps


def load_cmb_map(nside, sky="c1d0s0"):
    """Load cached CMB-only maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    sky : str, optional
        Sky model preset string (default: "c1d0s0").

    Returns
    -------
    ndarray
        CMB map with shape (3, n_pix) for Stokes I, Q, U, or zeros if no CMB.

    Raises
    ------
    FileNotFoundError
        If cache file does not exist.
    """
    # Define cache file path
    cache_dir = "freq_maps_cache"
    preset_strings = [sky[i : i + 2] for i in range(0, len(sky), 2)]
    cmb_template = None
    for strr in preset_strings:
        if strr.startswith("c"):
            cmb_template = strr
            break
    if cmb_template is None:
        npix = 12 * nside**2
        return np.zeros((3, npix))
    else:
        cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{cmb_template}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                freq_maps = pickle.load(f)
            print(f"Loaded freq_maps for nside {nside} from cache.")
        else:
            raise FileNotFoundError(
                f"Cache file for freq_maps with nside {nside} not found.\n"
                f"Please generate it first by calling `generate_maps({nside})`."
            )

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        raise FileNotFoundError(
            f"Cache file for freq_maps with nside {nside} not found.\n"
            f"Please generate it first by calling `generate_maps({nside})`."
        )

    return freq_maps


def get_mixin_matrix_operator(params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0):
    """Construct mixing matrix operators for CMB and foregrounds.

    Parameters
    ----------
    params : dict
        Spectral parameters (temp_dust, beta_dust, beta_pl).
    patch_indices : dict
        Patch assignment indices for each parameter.
    nu : array_like
        Frequency array in GHz.
    sky : dict
        Sky component dictionary from FURAX.
    dust_nu0 : float
        Dust reference frequency in GHz.
    synchrotron_nu0 : float
        Synchrotron reference frequency in GHz.

    Returns
    -------
    tuple
        (MixingMatrixOperator with CMB, MixingMatrixOperator without CMB).
    """
    first_element = next(iter(sky.values()))
    size = first_element.shape[-1]
    in_structure = first_element.structure_for((size,))

    cmb = CMBOperator(nu, in_structure=in_structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=params["temp_dust"],
        temperature_patch_indices=patch_indices["temp_dust_patches"],
        beta=params["beta_dust"],
        beta_patch_indices=patch_indices["beta_dust_patches"],
        in_structure=in_structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=params["beta_pl"],
        beta_pl_patch_indices=patch_indices["beta_pl_patches"],
        in_structure=in_structure,
    )

    return MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron), MixingMatrixOperator(
        dust=dust, synchrotron=synchrotron
    )


def simulate_D_from_params(params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0):
    """Simulate observed frequency maps given spectral parameters.

    Parameters
    ----------
    params : dict
        Spectral parameters (temp_dust, beta_dust, beta_pl).
    patch_indices : dict
        Patch assignment indices for each parameter.
    nu : array_like
        Frequency array in GHz.
    sky : dict
        Sky component dictionary.
    dust_nu0 : float
        Dust reference frequency in GHz.
    synchrotron_nu0 : float
        Synchrotron reference frequency in GHz.

    Returns
    -------
    tuple
        (d, d_nocmb) where d includes CMB and d_nocmb excludes it.
    """
    A, A_nocmb = get_mixin_matrix_operator(
        params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0
    )
    d = A(sky)
    sky_no_cmb = sky.copy()
    sky_no_cmb.pop("cmb")
    d_nocmb = A_nocmb(sky_no_cmb)
    return d, d_nocmb


MASK_CHOICES = [
    "ALL",
    "GALACTIC",
    "GAL020_U",
    "GAL020_L",
    "GAL020",
    "GAL040_U",
    "GAL040_L",
    "GAL040",
    "GAL060_U",
    "GAL060_L",
    "GAL060",
]


def get_mask(mask_name="GAL020", nside=64):
    """Load and process galactic masks at specified resolution.

    Parameters
    ----------
    mask_name : str, optional
        Mask identifier (e.g., "GAL020", "GAL040", "GALACTIC") (default: "GAL020").
    nside : int, optional
        HEALPix resolution parameter (default: 64).

    Returns
    -------
    ndarray
        Boolean mask array where True indicates observed pixels.

    Raises
    ------
    FileNotFoundError
        If mask file for given nside does not exist.
    ValueError
        If mask_name is invalid.

    Notes
    -----
    Available mask choices: ALL, GALACTIC, GAL020, GAL040, GAL060, and
    their _U (upper) and _L (lower) hemisphere variants.
    """
    masks_file = f"masks/GAL_PlanckMasks_{nside}.npz"

    try:
        masks = np.load(masks_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"""
            Could not find masks file: {masks_file}.
            please run the downgrade script with nside={nside}
            """)

    if mask_name not in MASK_CHOICES:
        raise ValueError(f"Invalid mask name: {mask_name}. Choose from: {MASK_CHOICES}")

    npix = 12 * nside**2
    ones = np.ones(npix, dtype=bool)
    # Extract the masks (keys: "GAL020", "GAL040", "GAL060").
    mask_GAL020 = masks["GAL020"]
    mask_GAL040 = masks["GAL040"]
    mask_GAL060 = masks["GAL060"]

    mask_galactic = np.logical_and(ones, np.logical_not(mask_GAL060))
    mask_GAL060 = np.logical_and(mask_GAL060, np.logical_not(mask_GAL040))
    mask_GAL040 = np.logical_and(mask_GAL040, np.logical_not(mask_GAL020))

    # Determine the HEALPix resolution (nside) from one of the masks.
    nside = hp.get_nside(mask_GAL020)

    # Get pixel indices and corresponding angular coordinates (theta, phi) in radians.
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix)

    # Define upper and lower hemispheres based on theta.
    # (Assuming theta < pi/2 corresponds to b > 0, i.e. the "upper" hemisphere.)
    upper = theta < np.pi / 2
    lower = theta >= np.pi / 2

    zones = {}

    zones["ALL"] = ones
    # --- Define Zones ---
    # GAL020 Upper ring and lower ring
    zones["GAL020_U"] = np.logical_and(mask_GAL020, upper)
    zones["GAL020_L"] = np.logical_and(mask_GAL020, lower)
    zones["GAL020"] = mask_GAL020
    # GAL040 Upper ring and lower ring
    zones["GAL040_U"] = np.logical_and(mask_GAL040, upper)
    zones["GAL040_L"] = np.logical_and(mask_GAL040, lower)
    zones["GAL040"] = mask_GAL040
    # GAL060 Upper ring and lower ring
    zones["GAL060_U"] = np.logical_and(mask_GAL060, upper)
    zones["GAL060_L"] = np.logical_and(mask_GAL060, lower)
    zones["GAL060"] = mask_GAL060
    # Galactic mask
    zones["GALACTIC"] = mask_galactic

    # Return the requested zone.
    return zones[mask_name]


def generate_needed_maps(nside_list=None, noise_list=None, instrument_name="LiteBIRD", sky_list=None):
    """Batch generate and cache all required frequency maps.

    Parameters
    ----------
    nside_list : list of int, optional
        HEALPix resolutions to generate (default: [4, 8, 32, 64, 128]).
    noise_list : list of bool, optional
        Noise configurations (default: [True, False]).
    instrument_name : str, optional
        Instrument configuration (default: "LiteBIRD").
    sky_list : list of str, optional
        Sky model presets (default: ["c1d0s0", "c1d1s1"]).

    Notes
    -----
    Generates full frequency maps, foreground-only maps, and CMB-only maps
    for all combinations of input parameters.
    """
    if nside_list is None:
        nside_list = [4, 8, 32, 64, 128]
    if noise_list is None:
        noise_list = [True, False]
    if sky_list is None:
        sky_list = ["c1d0s0", "c1d1s1"]

    for nside in nside_list:
        for noise in noise_list:
            for sky in sky_list:
                save_to_cache(nside, noise=noise, instrument_name=instrument_name, sky=sky)

    for sky in sky_list:
        for nside in nside_list:
            save_fg_map(nside, noise=False, instrument_name=instrument_name, sky=sky)
            save_cmb_map(nside, sky=sky)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cached frequency maps for CMB component separation"
    )
    parser.add_argument(
        "--nside",
        type=int,
        nargs="+",
        default=[4, 8, 32, 64, 128],
        help="HEALPix resolution(s) to generate maps for (default: 4 8 32 64 128)",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Generate only noise simulations (default: generate both noise and no-noise)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Generate only no-noise simulations (default: generate both noise and no-noise)",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="LiteBIRD",
        help="Instrument name (default: LiteBIRD)",
    )
    parser.add_argument(
        "--sky",
        type=str,
        nargs="+",
        default=["c1d0s0", "c1d1s1"],
        help="Sky model tag(s) (default: c1d0s0 c1d1s1)",
    )

    args = parser.parse_args()

    if args.noise and args.no_noise:
        parser.error("Cannot specify both --noise and --no-noise")

    if args.noise:
        noise_list = [True]
    elif args.no_noise:
        noise_list = [False]
    else:
        noise_list = [True, False]

    generate_needed_maps(
        nside_list=args.nside,
        noise_list=noise_list,
        instrument_name=args.instrument,
        sky_list=args.sky,
    )


if __name__ == "__main__":
    main()
