"""Data generation and instrument configuration for CMB component separation."""

from .generate_maps import (
    get_mixin_matrix_operator,
    load_cmb_map,
    load_fg_map,
    load_from_cache,
    save_cmb_map,
    save_fg_map,
    save_to_cache,
    simulate_D_from_params,
)
from .instruments import get_instrument
from .search_space import load_search_space, search_space_to_jax, dump_default_search_space, validate_search_space


__all__ = [
    "generate_maps",
    "get_mixin_matrix_operator",
    "load_cmb_map",
    "load_fg_map",
    "load_from_cache",
    "save_cmb_map",
    "save_fg_map",
    "save_to_cache",
    "simulate_D_from_params",
    "get_instrument",
    "load_search_space",
    "search_space_to_jax",
    "dump_default_search_space",
    "validate_search_space",
]