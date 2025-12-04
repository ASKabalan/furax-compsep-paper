import jax
import numpy as np
from furax.obs.stokes import StokesI, StokesIQU, StokesQU


def expand_stokes(stokes_map):
    """Promote a StokesI or StokesQU instance to StokesIQU.

    Parameters
    ----------
    stokes_map : StokesI | StokesQU | StokesIQU
        Input Stokes data structure produced by the component separation pipeline.

    Returns
    -------
    StokesIQU
        Stokes object with I, Q, U components explicitly populated.
    """
    if isinstance(stokes_map, StokesIQU):
        return stokes_map

    zeros = np.zeros(shape=stokes_map.shape, dtype=stokes_map.dtype)

    if isinstance(stokes_map, StokesI):
        return StokesIQU(stokes_map, zeros, zeros)
    elif isinstance(stokes_map, StokesQU):
        return StokesIQU(zeros, stokes_map.q, stokes_map.u)


def filter_constant_param(input_dict, indx):
    """Extract a specific entry from a tree of arrays.

    Parameters
    ----------
    input_dict : Mapping[str, ndarray]
        Tree containing clustering results indexed by realization.
    indx : int
        Index to select along the leading dimension of each array.

    Returns
    -------
    Mapping[str, ndarray]
        Tree with arrays sliced at the requested index.
    """
    return jax.tree.map(lambda x: x[indx], input_dict)


def index_run_data(run_data, run_index):
    """Select the requested run across cached result arrays.

    Cached quantities whose keys start with ``W_D_FG_`` or ``CL_BB_SUM_`` already
    correspond to expensive per-run products, so they are returned unchanged.

    Parameters
    ----------
    run_data : Mapping[str, ndarray]
        Output dictionary for a single clustering configuration.
    run_index : int
        Index of the noise realization to extract.

    Returns
    -------
    Mapping[str, ndarray]
        Same structure as ``run_data`` with realizations sliced on demand.
    """

    def should_index(path, value):
        key = path[-1].key if path else None
        if key and (key.startswith("W_D_FG_") or key.startswith("CL_BB_SUM_")):
            return value
        return value[run_index]

    return jax.tree_util.tree_map_with_path(should_index, run_data)


def sort_results(results, key):
    """Sort a result tree by an array value.

    Parameters
    ----------
    results : Mapping[str, ndarray]
        Container produced by grid search evaluations.
    key : str
        Key whose values define the ordering.

    Returns
    -------
    Mapping[str, ndarray]
        Tree with entries reordered consistently.
    """
    indices = np.argsort(results[key])
    return jax.tree.map(lambda x: x[indices], results)


def params_to_maps(run_data, previous_mask_size):
    """Convert per-cluster parameter arrays to HEALPix maps.

    Parameters
    ----------
    run_data : Mapping[str, ndarray]
        Output of a clustering run containing parameters and patch indices.
    previous_mask_size : Mapping[str, int]
        Offsets that keep cluster labels unique across disjoint sky regions.

    Returns
    -------
    tuple
        (params, patches, updated_offsets) where params are maps of mean
        spectral parameters, patches contains normalized cluster indices and
        updated_offsets holds cumulative label offsets per parameter.
    """
    B_d_patches = run_data["beta_dust_patches"]
    T_d_patches = run_data["temp_dust_patches"]
    B_s_patches = run_data["beta_pl_patches"]

    cmb_variance = run_data["value"]
    indx = np.argmin(cmb_variance)

    B_d = run_data["beta_dust"]
    T_d = run_data["temp_dust"]
    B_s = run_data["beta_pl"]

    B_d = B_d[indx][B_d_patches]
    T_d = T_d[indx][T_d_patches]
    B_s = B_s[indx][B_s_patches]

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
