# FGBUSTER IMPORTS

from fgbuster import (
    get_observation,
    get_instrument,
)


import os
import pickle

from furax.landscapes import StokesPyTree
from furax.operators.seds import CMBOperator, DustOperator, SynchrotronOperator, MixingMatrixOperator


def save_to_cache(nside, noise=False, instrument_name='LiteBIRD' , sky='c1d0s0'):
    instrument = get_instrument(instrument_name)
    # Define cache file path
    cache_dir = 'freq_maps_cache'
    os.makedirs(cache_dir, exist_ok=True)
    noise_str = 'noise' if noise else 'no_noise'
    cache_file = os.path.join(cache_dir, f'freq_maps_nside_{nside}_{noise_str}_{sky}.pkl')

    # Check if file exists, load if it does, otherwise create and save it
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            freq_maps = pickle.load(f)
        print(f'Loaded freq_maps for nside {nside} from cache.')
    else:
        # Generate freq_maps if not already cached
        freq_maps = get_observation(instrument, sky, nside=nside, noise=noise)

        # Save freq_maps to the cache
        with open(cache_file, 'wb') as f:
            pickle.dump(freq_maps, f)
        print(f'Generated and saved freq_maps for nside {nside}.')


def load_from_cache(nside, noise=False, instrument_name='LiteBIRD', sky='c1d0s0'):
    # Define cache file path
    instrument = get_instrument(instrument_name)
    noise_str = 'noise' if noise else 'no_noise'
    cache_dir = 'freq_maps_cache'
    cache_file = os.path.join(cache_dir, f'freq_maps_nside_{nside}_{noise_str}_{sky}.pkl')

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            freq_maps = pickle.load(f)
        print(f'Loaded freq_maps for nside {nside} from cache.')
    else:
        raise FileNotFoundError(
            f'Cache file for freq_maps with nside {nside} not found.\n'
            f'Please generate it first by calling `generate_maps({nside})`.'
        )

    return instrument['frequency'].values, freq_maps


def simulate_D_from_params(params, patch_indices, nu, sky, stokes, dust_nu0, synchrotron_nu0):
    size = next(iter(sky.values())).shape
    in_structure = StokesPyTree.class_for(stokes).structure_for(size)

    cmb = CMBOperator(nu, in_structure=in_structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=params['temp_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta=params['beta_dust'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=in_structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=params['beta_pl'],
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=in_structure,
    )

    A = MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)
    d = A(sky)

    return d
