# IMPORTS

import argparse
import os
import pickle
from pathlib import Path

import healpy as hp
import jax.random as jr
import numpy as np
from furax._instruments.sky import get_observation, get_sky
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    SynchrotronOperator,
)

from furax_cs.data.instruments import get_instrument


def save_to_cache(nside, noise_ratio=0.0, instrument_name="LiteBIRD", sky="c1d0s0", key=None):
    """Generate and cache frequency maps for component separation.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise_ratio : float, optional
        Noise level ratio (0.0 = no noise, 1.0 = 100% noise) (default: 0.0).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string (e.g., "c1d0s0" for CMB only) (default: "c1d0s0").
    key : PRNGKeyArray, optional
        JAX random key for noise generation (default: None, uses key(0)).

    Returns
    -------
    tuple
        (frequencies, freq_maps) where frequencies are in GHz and freq_maps
        have shape (n_freq, 3, n_pix) for Stokes I, Q, U.
    """
    if key is None:
        key = jr.PRNGKey(0)

    instrument = get_instrument(instrument_name)
    # Define cache file path
    cache_dir = "freq_maps_cache"
    os.makedirs(cache_dir, exist_ok=True)
    noise_str = f"noise_{int(noise_ratio * 100)}" if noise_ratio > 0 else "no_noise"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists, load if it does, otherwise create and save it
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache with noise_ratio {noise_ratio}.")
    else:
        # Generate freq_maps if not already cached
        stokes_obs = get_observation(
            instrument,
            nside=nside,
            tag=sky,
            noise_ratio=noise_ratio,
            key=key,
            stokes_type="IQU",
            unit="uK_CMB",
        )

        # Convert Stokes PyTree to numpy array (n_freq, 3, n_pix)
        freq_maps = np.stack(
            [np.array(stokes_obs.i), np.array(stokes_obs.q), np.array(stokes_obs.u)], axis=1
        )

        # Save freq_maps to the cache
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        print(f"Generated and saved freq_maps for nside {nside}.")
    return np.array(instrument.frequency), freq_maps


def load_from_cache(nside, noise_ratio=0.0, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Load cached frequency maps from disk.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise_ratio : float, optional
        Noise level ratio (0.0 = no noise, 1.0 = 100% noise) (default: 0.0).
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
    noise_str = f"noise_{int(noise_ratio * 100)}" if noise_ratio > 0 else "no_noise"
    cache_dir = "freq_maps_cache"
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}_{noise_str}_{sky}.pkl")

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        raise FileNotFoundError(
            f"Cache file for freq_maps with nside {nside} and noise_ratio {noise_ratio} not found.\n"
            f"Please generate it first by calling `generate_data --nside {nside}`."
        )

    return np.array(instrument.frequency), freq_maps


def strip_cmb_tag(sky_string):
    """Removes the 'cX' tag from the sky string."""
    tags = [sky_string[i : i + 2] for i in range(0, len(sky_string), 2)]
    fg_tags = [tag for tag in tags if not tag.startswith("c")]
    return "".join(fg_tags)


def save_fg_map(nside, noise_ratio=0.0, instrument_name="LiteBIRD", sky="c1d0s0", key=None):
    """Generate and cache foreground-only frequency maps (CMB excluded).

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise_ratio : float, optional
        Noise level ratio (0.0 = no noise, 1.0 = 100% noise) (default: 0.0).
    instrument_name : str, optional
        Instrument configuration name (default: "LiteBIRD").
    sky : str, optional
        Sky model preset string, CMB component automatically removed (default: "c1d0s0").
    key : PRNGKeyArray, optional
        JAX random key for noise generation (default: None).

    Returns
    -------
    tuple
        (frequencies, freq_maps) containing only foreground contributions.
    """
    print(
        f"Generating fg map for nside {nside}, noise_ratio {noise_ratio}, instrument {instrument_name}"
    )
    stripped_sky = strip_cmb_tag(sky)
    return save_to_cache(
        nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=stripped_sky, key=key
    )


def load_fg_map(nside, noise_ratio=0.0, instrument_name="LiteBIRD", sky="c1d0s0"):
    """Load cached foreground-only frequency maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    noise_ratio : float, optional
        Noise level ratio (0.0 = no noise, 1.0 = 100% noise) (default: 0.0).
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
    return load_from_cache(
        nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=stripped_sky
    )


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
                f"Please generate it first by calling `generate_data --nside {nside}`."
            )

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        raise FileNotFoundError(
            f"Cache file for freq_maps with nside {nside} not found.\n"
            f"Please generate it first by calling `generate_data --nside {nside}`."
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


def sanitize_mask_name(mask_expr: str) -> str:
    """Convert mask expression to valid folder name.

    Parameters
    ----------
    mask_expr : str
        Mask expression potentially containing + (union) or - (subtract) operators.

    Returns
    -------
    str
        Sanitized folder name with operators replaced by descriptive names.

    Examples
    --------
    >>> sanitize_mask_name("GAL020+GAL040")
    'GAL020_UNION_GAL040'
    >>> sanitize_mask_name("ALL-GALACTIC")
    'ALL_SUBTRACT_GALACTIC'
    """
    sanitized = mask_expr.replace("+", "_UNION_").replace("-", "_SUBTRACT_")
    return sanitized


def parse_mask_expression(expr: str, nside: int) -> np.ndarray:
    """Parse and evaluate boolean mask expressions.

    Supports left-to-right evaluation of expressions with + (union) and - (subtraction)
    operators. Does not support parentheses.

    Parameters
    ----------
    expr : str
        Mask expression with optional boolean operators.
        Examples: "GAL020+GAL040", "ALL-GALACTIC", "GAL020+GAL040-GALACTIC"
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    ndarray
        Boolean mask array where True indicates observed pixels.

    Raises
    ------
    ValueError
        If expression contains invalid mask names or syntax.

    Examples
    --------
    >>> mask = parse_mask_expression("GAL020+GAL040", nside=64)
    >>> mask = parse_mask_expression("ALL-GALACTIC", nside=64)
    """
    # Tokenize the expression while preserving operators
    tokens = []
    current_token = ""

    for char in expr:
        if char in ["+", "-"]:
            if current_token:
                tokens.append(current_token.strip())
                current_token = ""
            tokens.append(char)
        else:
            current_token += char

    if current_token:
        tokens.append(current_token.strip())

    if not tokens:
        raise ValueError(f"Empty mask expression: {expr}")

    # Validate that we have alternating mask names and operators
    if len(tokens) == 1:
        # Single mask, no operators
        mask_name = tokens[0]
        if mask_name not in MASK_CHOICES:
            raise ValueError(
                f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                f"Choose from: {MASK_CHOICES}"
            )
        return get_mask(mask_name, nside)

    # Multiple tokens - evaluate left to right
    if len(tokens) % 2 == 0:
        raise ValueError(
            f"Invalid expression syntax: {expr}. Expected format: MASK [+/-] MASK [+/-] MASK ..."
        )

    # Start with first mask
    result = None
    i = 0

    while i < len(tokens):
        if i == 0:
            # First token must be a mask name
            mask_name = tokens[i]
            if mask_name not in MASK_CHOICES:
                raise ValueError(
                    f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                    f"Choose from: {MASK_CHOICES}"
                )
            result = get_mask(mask_name, nside)
            i += 1
        else:
            # Even indices are operators, odd are mask names
            operator = tokens[i]
            if operator not in ["+", "-"]:
                raise ValueError(
                    f"Expected operator (+ or -) at position {i} in expression '{expr}', "
                    f"got '{operator}'"
                )

            if i + 1 >= len(tokens):
                raise ValueError(f"Operator '{operator}' at end of expression '{expr}'")

            mask_name = tokens[i + 1]
            if mask_name not in MASK_CHOICES:
                raise ValueError(
                    f"Invalid mask name '{mask_name}' in expression '{expr}'. "
                    f"Choose from: {MASK_CHOICES}"
                )

            next_mask = get_mask(mask_name, nside)

            if operator == "+":
                # Union
                result = np.logical_or(result, next_mask)
            elif operator == "-":
                # Subtraction
                result = np.logical_and(result, np.logical_not(next_mask))

            i += 2

    return result


def _get_or_generate_mask_file(nside: int) -> Path:
    """Get path to mask file, generating it from 2048 source if needed.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    Path
        Path to the mask file.
    """
    mask_dir = Path(__file__).parent / "masks"
    mask_file = mask_dir / f"GAL_PlanckMasks_{nside}.npz"

    if mask_file.exists():
        return mask_file

    # Generate from 2048 source
    source_file = mask_dir / "GAL_PlanckMasks_2048.npz"
    masks_2048 = np.load(source_file)

    downgraded = {}
    for key in masks_2048.files:
        downgraded[key] = hp.ud_grade(masks_2048[key] * 1.0, nside).astype(np.uint8)

    np.savez(mask_file, **downgraded)
    print(f"Generated and cached mask for nside {nside}")

    return mask_file


def get_mask(mask_name="GAL020", nside=64):
    """Load and process galactic masks at specified resolution.

    Parameters
    ----------
    mask_name : str, optional
        Mask identifier (e.g., "GAL020", "GAL040", "GALACTIC") or boolean expression
        (e.g., "GAL020+GAL040", "ALL-GALACTIC") (default: "GAL020").
    nside : int, optional
        HEALPix resolution parameter (default: 64).

    Returns
    -------
    ndarray
        Boolean mask array where True indicates observed pixels.

    Raises
    ------
    ValueError
        If mask_name is invalid.

    Notes
    -----
    Available mask choices: ALL, GALACTIC, GAL020, GAL040, GAL060, and
    their _U (upper) and _L (lower) hemisphere variants.

    Boolean operations are supported:
    - Use + for union (logical OR)
    - Use - for subtraction (logical AND NOT)
    - Expressions are evaluated left-to-right
    - Examples: "GAL020+GAL040", "ALL-GALACTIC", "GAL020+GAL040-GALACTIC"

    Masks are automatically generated and cached on first call for each nside.
    """
    # Check if mask_name contains boolean operators
    if "+" in mask_name or "-" in mask_name:
        return parse_mask_expression(mask_name, nside)

    masks_file = _get_or_generate_mask_file(nside)
    masks = np.load(masks_file)

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


def generate_needed_maps(
    nside_list=None, noise_ratio_list=None, instrument_name="LiteBIRD", sky_list=None
):
    """Batch generate and cache all required frequency maps.

    Parameters
    ----------
    nside_list : list of int, optional
        HEALPix resolutions to generate (default: [4, 8, 32, 64, 128]).
    noise_ratio_list : list of float, optional
        Noise ratio configurations (default: [0.0, 1.0]).
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
    if noise_ratio_list is None:
        noise_ratio_list = [0.0, 1.0]
    if sky_list is None:
        sky_list = ["c1d0s0", "c1d1s1"]

    for nside in nside_list:
        for noise_ratio in noise_ratio_list:
            for sky in sky_list:
                save_to_cache(
                    nside, noise_ratio=noise_ratio, instrument_name=instrument_name, sky=sky
                )

    for sky in sky_list:
        for nside in nside_list:
            save_fg_map(nside, noise_ratio=0.0, instrument_name=instrument_name, sky=sky)
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
        "--noise-ratio",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Noise ratio level(s) to generate (0.0=no noise, 1.0=100%% noise, default: 0.0 1.0)",
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

    generate_needed_maps(
        nside_list=args.nside,
        noise_ratio_list=args.noise_ratio,
        instrument_name=args.instrument,
        sky_list=args.sky,
    )


if __name__ == "__main__":
    main()
