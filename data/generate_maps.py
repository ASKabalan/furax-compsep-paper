# FGBUSTER IMPORTS

import os
import pickle
from pathlib import Path

import healpy as hp
import numpy as np
from fgbuster import (
    get_instrument,
    get_observation,
)
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    SynchrotronOperator,
)


def save_to_cache(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
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
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        # Generate freq_maps if not already cached
        freq_maps = get_observation(instrument, sky, nside=nside, noise=noise)

        # Save freq_maps to the cache
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        print(f"Generated and saved freq_maps for nside {nside}.")
    return instrument["frequency"].values, freq_maps


def load_from_cache(nside, noise=False, instrument_name="LiteBIRD", sky="c1d0s0"):
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


def get_mixin_matrix_operator(params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0):
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

    return MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)


def simulate_D_from_params(params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0):
    A = get_mixin_matrix_operator(params, patch_indices, nu, sky, dust_nu0, synchrotron_nu0)
    d = A(sky)

    return d


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
    current_dir = Path(__file__).parent
    masks_file = f"{current_dir}/masks/GAL_PlanckMasks_{nside}.npz"

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
